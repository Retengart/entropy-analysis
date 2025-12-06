#!/usr/bin/env python3
"""
Скрипт для фильтрации файла lermontov.txt для анализа с разделителями.
Разделяет стихи разделителями ===== и удаляет заголовки страниц, номера страниц и примечания редактора.
"""

import re
from typing import List, Tuple, Set


def read_file_lines(filepath: str) -> List[str]:
    """Читает файл построчно."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()


def extract_poem_titles_from_toc(lines: List[str], toc_end_line: int) -> Set[str]:
    """
    Извлекает названия стихов из оглавления.
    """
    titles = set()
    toc_lines = lines[:toc_end_line]
    
    # Паттерны для пропуска:
    skip_patterns = [
        r'^Содержание$',
        r'^Стихотворения \d{4}',
        r'^Примечания к стихотворениям',
        r'^М\. Ю\. Лермонтов\.',
        r'^\d+$',  # Только цифры (номера страниц)
        r'^\s*$',  # Пустые строки
    ]
    
    for line in toc_lines:
        line = line.strip()
        if not line:
            continue
        
        # Пропускаем строки, соответствующие паттернам
        should_skip = False
        for pattern in skip_patterns:
            if re.match(pattern, line):
                should_skip = True
                break
        
        if should_skip:
            continue
        
        # Извлекаем название стиха (до номера страницы, если есть)
        match = re.match(r'^(.+?)\s+\d+$', line)
        if match:
            title = match.group(1).strip()
            if title:
                titles.add(title)
        elif line and not line.isdigit():
            # Если нет номера страницы, берем всю строку
            titles.add(line)
    
    return titles


def normalize_title(title: str) -> str:
    """Нормализует название для сравнения."""
    title = re.sub(r'\s+', ' ', title.strip())
    title = re.sub(r'<[^>]+>', '', title)
    return title.lower()


def is_page_number(line: str, next_line: str) -> bool:
    """
    Проверяет, является ли строка номером страницы.
    Номер страницы - это строка, состоящая только из цифр,
    которая находится непосредственно перед заголовком страницы.
    """
    page_header = "М. Ю. Лермонтов. «Полное собрание стихотворений»"
    return (re.match(r'^\d+$', line.strip()) and 
            next_line.strip() == page_header)


def is_notes_section(line: str) -> bool:
    """Проверяет, является ли строка началом раздела примечаний."""
    return bool(re.match(r'^Примечания к стихотворениям', line.strip()))


def find_poem_boundaries(lines: List[str], start_line: int, poem_titles: Set[str]) -> List[Tuple[int, int]]:
    """
    Находит границы стихов в тексте.
    Возвращает список кортежей (start, end) для каждого стиха.
    """
    boundaries = []
    current_start = None
    current_poem_title = None
    in_notes_section = False
    
    page_header = "М. Ю. Лермонтов. «Полное собрание стихотворений»"
    section_pattern = re.compile(r'^Стихотворения \d{4}')
    
    # Нормализуем названия стихов для сравнения
    normalized_titles = {normalize_title(t): t for t in poem_titles}
    
    i = start_line
    while i < len(lines):
        line = lines[i].strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        
        # Проверяем, не начался ли раздел примечаний
        if is_notes_section(line):
            # Если был текущий стих, завершаем его перед примечаниями
            if current_start is not None:
                boundaries.append((current_start, i))
                current_start = None
                current_poem_title = None
            # Пропускаем весь раздел примечаний до следующего заголовка страницы
            in_notes_section = True
            i += 1
            continue
        
        # Если мы в разделе примечаний, пропускаем до следующего заголовка страницы
        if in_notes_section:
            if line == page_header:
                # Заголовок страницы означает конец примечаний и начало нового раздела стихов
                in_notes_section = False
            i += 1
            continue
        
        # Начало страницы
        if line == page_header:
            i += 1
            # После заголовка страницы может быть название группы или название стиха
            if i < len(lines):
                next_line = lines[i].strip()
                
                # Пропускаем служебные строки в начале файла
                if next_line in ["Михаил Юрьевич Лермонтов", "Полное собрание стихотворений"]:
                    i += 1
                    continue
                
                # Если это название группы стихов, пропускаем его
                if section_pattern.match(next_line):
                    i += 1
                    # Следующая строка должна быть названием стиха или началом стиха
                    if i < len(lines):
                        next_line = lines[i].strip()
                
                # Проверяем, является ли следующая строка названием стиха
                if next_line and not is_page_number(next_line, lines[i + 1].strip() if i + 1 < len(lines) else ""):
                    normalized = normalize_title(next_line)
                    # Если это название стиха из оглавления
                    if normalized in normalized_titles:
                        # Если это НОВОЕ название стиха (отличается от текущего)
                        if current_poem_title != normalized:
                            # Если был предыдущий стих, завершаем его
                            if current_start is not None:
                                boundaries.append((current_start, i - 1))
                            # Начинаем новый стих
                            current_start = i
                            current_poem_title = normalized
                        # Если это то же название, что и текущее - это продолжение стиха
                    elif current_start is None:
                        # Если это не название стиха, но мы еще не начали стих,
                        # это может быть начало первого стиха (без названия в оглавлении)
                        current_start = i
            i += 1
            continue
        
        # Номер страницы (только цифры перед заголовком страницы) - конец страницы
        if is_page_number(line, next_line):
            i += 1
            continue
        
        # Если мы еще не начали стих и это не пустая строка
        if current_start is None and line:
            # Пропускаем строки типа "Стихотворения 1828 года"
            if section_pattern.match(line):
                i += 1
                continue
            # Пропускаем служебные строки
            if line in ["Михаил Юрьевич Лермонтов", "Полное собрание стихотворений"]:
                i += 1
                continue
            # Начинаем новый стих (первый стих в файле)
            current_start = i
            # Пытаемся определить название стиха
            normalized = normalize_title(line)
            if normalized in normalized_titles:
                current_poem_title = normalized
        
        i += 1
    
    # Если остался незавершенный стих
    if current_start is not None:
        boundaries.append((current_start, len(lines)))
    
    return boundaries


def filter_file(input_file: str, output_file: str):
    """Основная функция фильтрации файла."""
    print(f"Чтение файла {input_file}...")
    lines = read_file_lines(input_file)
    
    # Оглавление заканчивается примерно на строке 628 (последняя строка "Примечания к стихотворениям разных лет")
    # Находим последнюю строку оглавления
    toc_end = 0
    for i, line in enumerate(lines):
        if re.match(r'^Примечания к стихотворениям разных лет', line.strip()):
            toc_end = i + 1
            break
    
    if toc_end == 0:
        # Если не нашли, используем приблизительное значение
        toc_end = 630
    
    print(f"Оглавление заканчивается на строке {toc_end}")
    
    print("Извлечение названий стихов из оглавления...")
    poem_titles = extract_poem_titles_from_toc(lines, toc_end)
    print(f"Найдено {len(poem_titles)} названий стихов в оглавлении")
    
    # Текст стихов начинается после оглавления
    text_start = toc_end
    
    print("Поиск границ стихов в тексте...")
    boundaries = find_poem_boundaries(lines, text_start, poem_titles)
    print(f"Найдено {len(boundaries)} стихов в тексте")
    
    # Записываем отфильтрованный файл
    print(f"Запись результата в {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        page_header = "М. Ю. Лермонтов. «Полное собрание стихотворений»"
        section_pattern = re.compile(r'^Стихотворения \d{4}')
        in_notes = False
        
        for i, (start, end) in enumerate(boundaries):
            # Записываем разделитель перед каждым стихом (кроме первого)
            if i > 0:
                f.write("======\n")
            
            # Извлекаем текст стиха
            poem_lines = []
            # Сбрасываем флаг примечаний для каждого стиха
            in_notes_local = False
            
            for j in range(start, end):
                line = lines[j].strip()
                next_line = lines[j + 1].strip() if j + 1 < len(lines) else ""
                
                # Пропускаем заголовки страниц
                if line == page_header:
                    continue
                
                # Пропускаем номера страниц (только если следующая строка - заголовок страницы)
                if is_page_number(line, next_line) or (re.match(r'^\d+$', line) and j == end - 1):
                    continue
                
                # Пропускаем заголовки разделов типа "Стихотворения 1828 года"
                if section_pattern.match(line):
                    continue
                
                # Пропускаем разделы примечаний (если стих начинается с примечаний, пропускаем весь стих)
                if is_notes_section(line):
                    # Это раздел примечаний - пропускаем весь стих
                    poem_lines = []
                    break
                
                poem_lines.append(line)
            
            # Удаляем пустые строки в начале и конце
            while poem_lines and not poem_lines[0]:
                poem_lines.pop(0)
            while poem_lines and not poem_lines[-1]:
                poem_lines.pop()
            
            # Записываем стих
            if poem_lines:
                f.write('\n'.join(poem_lines))
                f.write('\n')
    
    print("Готово!")


if __name__ == "__main__":
    input_file = "science/lermontov.txt"
    output_file = "science/lermontov_filtered.txt"
    filter_file(input_file, output_file)

