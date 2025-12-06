#!/usr/bin/env python3
"""
Удаление примечаний редактора на основе оглавления.
В оглавлении указаны номера страниц, где начинаются разделы примечаний.
"""

import re


def find_notes_sections_in_toc(toc_lines):
    """Находит разделы примечаний в оглавлении."""
    notes_sections = []
    for line in toc_lines:
        if 'Примечания к стихотворениям' in line:
            match = re.search(r'(\d+)$', line.strip())
            if match:
                page_num = int(match.group(1))
                notes_sections.append(page_num)
    return notes_sections


def remove_notes_from_filtered_file(input_file: str, output_file: str):
    """
    Удаляет примечания из отфильтрованного файла.
    Примечания обычно идут после названия стиха и содержат:
    - "Печатается по..."
    - "Датируется..."
    - "Впервые опубликовано..."
    - Архивные ссылки (ИРЛИ, ГИМ, ГПБ, л., стр., т., №)
    - Ссылки на издания (Соч., под ред., записках, запискам)
    """
    print(f"Чтение файла {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Разделяем на стихи
    poems = content.split('======')
    
    cleaned_poems = []
    total_removed = 0
    
    # Паттерны для определения примечаний
    note_start_patterns = [
        r'^Печатается по',
        r'^Датируется',
        r'^Впервые',
        r'^Автограф',
        r'^Имеется',
        r'^В литературе',
        r'^Под заглавием',
        r'^Стихи \d+',
        r'^В \d+ году',
        r'^Строка.*исправляется',
        r'^Копия',
        r'^Черновой',
        r'^Некоторые стихи',
        r'^В этом стихотворении',
        r'^По словам',
        r'^В «Стихотворениях»',
        r'^В автографе',
        r'^Следующие.*стихотворений',
        r'^В рукописи',
        r'^В стихе \d+',
        r'^чтение стиха',
        r'^\d+.*стр\.',  # Строки, начинающиеся с цифр и содержащие ссылки
        r'^\(.*копия',  # Строки, начинающиеся со скобок и содержащие "копия"
        r'^Е\. А\. Сушкова',  # Конкретные имена в примечаниях
    ]
    
    for poem in poems:
        lines = poem.strip().split('\n')
        cleaned_lines = []
        in_note_block = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Пропускаем пустые строки
            if not line_stripped:
                if in_note_block:
                    continue
                cleaned_lines.append('')
                continue
            
            # Проверяем, является ли строка началом примечания
            is_note_start = False
            for pattern in note_start_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_note_start = True
                    break
            
            # Также проверяем строки с архивными ссылками или ссылками на издания
            if not is_note_start:
                has_archive_refs = bool(re.search(r'ИРЛИ|ГИМ|ГПБ|л\. \d+|стр\. \d+|т\. \d+|№ \d+|оп\.|ф\.', line_stripped))
                has_publication_refs = bool(re.search(r'Соч\.|под ред\.|записках|запискам|Библиогр\.|Отеч\.|Русск\.|Лит\.|Стихотворениях.*Лермонтова', line_stripped))
                
                # Длинные строки с ссылками - это примечания
                if (has_archive_refs or has_publication_refs) and len(line_stripped) > 60:
                    is_note_start = True
                
                # Строки с пометами и примечаниями
                if re.search(r'Написано на стенах|Перед второй частью|помета|Есть предположение|Произведение является|мому.*навеяны|рению\.$|стихотворению\)|отношение к этому стихотворению|озвученная|заглавленная', line_stripped, re.IGNORECASE):
                    is_note_start = True
                
                # Строки, начинающиеся с маленькой буквы и заканчивающиеся точкой (короткие примечания)
                if line_stripped[0].islower() and line_stripped.endswith('.') and len(line_stripped) < 50:
                    if re.search(r'датировку|окна|рению|стихотворению', line_stripped):
                        is_note_start = True
                
                # Строки с названиями мест (монастырь, города) в контексте примечаний
                if re.search(r'Воскресенск|монастырь.*верстах|иначе назывался', line_stripped, re.IGNORECASE):
                    is_note_start = True
                
                # Строки с пояснениями (Оссиан, легендарный, предания)
                if re.search(r'Оссиан.*легендарный|по семейным преданиям|Возможно.*является|новый смысл\.$|в скобках\.$|описание сей могилы|Обращено.*вероятности|набросками.*строф.*написанными', line_stripped, re.IGNORECASE):
                    is_note_start = True
            
            if is_note_start:
                in_note_block = True
                total_removed += 1
                continue
            
            # Если мы в блоке примечаний, проверяем продолжение
            if in_note_block:
                # Продолжение примечания определяется по:
                # 1. Начинается с маленькой буквы после примечания
                # 2. Содержит архивные/издательские ссылки
                # 3. Длинная строка (>80 символов)
                # 4. Начинается с цифр/скобок и содержит ссылки
                
                is_continuation = False
                
                # Строки, начинающиеся с цифр или скобок
                if re.match(r'^\d+|^\(', line_stripped):
                    if re.search(r'стр\.|т\.|№|Соч\.|под ред\.|ИРЛИ|ГИМ|ГПБ', line_stripped):
                        is_continuation = True
                
                # Строки с маленькой буквы после примечания
                if line_stripped[0].islower() and len(line_stripped) > 30:
                    if re.search(r'стр\.|т\.|№|Соч\.|под ред\.|стихотворению|датировку|монастырь|после стихотворения|в своих.*Записках', line_stripped):
                        is_continuation = True
                
                # Длинные строки с ссылками
                if len(line_stripped) > 80:
                    if re.search(r'ИРЛИ|ГИМ|ГПБ|Соч\.|под ред\.|записках|запискам|стр\.|т\.|№', line_stripped):
                        is_continuation = True
                
                # Короткие строки, заканчивающиеся точкой/скобкой после примечания
                if len(line_stripped) < 80 and re.search(r'[\.\)»]$', line_stripped):
                    if re.search(r'стихотворению|датировку|монастырь|окна|вероятно|осталась', line_stripped):
                        is_continuation = True
                
                # Строки с фразами типа "Произведение является..."
                if re.search(r'Произведение является|озвученная|отношение к этому стихотворению', line_stripped, re.IGNORECASE):
                    is_continuation = True
                
                # Строки с пометами в кавычках после стихов
                if re.search(r'Написано на стенах|Перед второй частью|помета|Есть предположение|Произведение является|мому.*навеяны|рению\.$|стихотворению\)|отношение к этому стихотворению|озвученная|заглавленная|с заглавием помета', line_stripped, re.IGNORECASE):
                    is_continuation = True
                
                # Строки, начинающиеся с маленькой буквы и заканчивающиеся точкой
                if line_stripped[0].islower() and line_stripped.endswith('.') and len(line_stripped) < 50:
                    if re.search(r'датировку|окна|рению|стихотворению|вероятно', line_stripped):
                        is_continuation = True
                
                # Строки с названиями мест (монастырь, города) в контексте примечаний
                if re.search(r'Воскресенск|монастырь.*верстах|иначе назывался', line_stripped, re.IGNORECASE):
                    is_continuation = True
                
                if is_continuation:
                    total_removed += 1
                    continue
                else:
                    # Это начало нового текста стиха
                    in_note_block = False
            
            cleaned_lines.append(line_stripped)
        
        # Удаляем пустые строки в конце
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        # Если остался текст, добавляем стих
        if cleaned_lines:
            cleaned_poems.append('\n'.join(cleaned_lines))
    
    # Записываем очищенный файл
    print(f"\nУдалено строк с примечаниями: {total_removed}")
    print(f"Стихов до очистки: {len(poems)}")
    print(f"Стихов после очистки: {len(cleaned_poems)}")
    
    print(f"\nЗапись очищенного файла в {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, poem in enumerate(cleaned_poems):
            if i > 0:
                f.write("======\n")
            f.write(poem)
            f.write('\n')
    
    print("Готово!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) >= 3 else input_file
    else:
        input_file = "science/lermontov_filtered.txt"
        output_file = "science/lermontov_filtered.txt"
    remove_notes_from_filtered_file(input_file, output_file)

