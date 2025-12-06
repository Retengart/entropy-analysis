#!/usr/bin/env python3
"""
Скрипт для удаления сносок и примечаний из отфильтрованного файла pushkin_filtered.txt
"""

import re


def remove_footnotes(input_file: str, output_file: str):
    """Удаляет сноски и примечания из файла."""
    print(f"Чтение файла {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_line_count = len(lines)
    removed_lines = []
    processed_lines = []
    
    # Паттерны для определения сносок
    # 1. Отдельные строки, состоящие только из текста в скобках
    standalone_brackets = re.compile(r'^\s*\([^)]+\)\s*$')
    
    # 2. Отдельные строки с номерами сносок (только цифры)
    footnote_number = re.compile(r'^\s*\d+\s*$')
    
    # 3. Языковые пометки в конце строк: (польск.), (лат.), (фр.) и т.д.
    language_note = re.compile(r'\([а-яё]+\.\)\s*$')
    
    # 4. Примечания в конце строк: (отрывок), (сказка), (притча)
    note_in_brackets = re.compile(r'\([а-яё]+\)\s*$')
    
    # 5. Ссылки: (см. стр.), (прим. ред.)
    reference_note = re.compile(r'\(см\.|\(прим\.')
    
    # 6. Даты в скобках в конце: (1815), (4 мая)
    date_in_brackets = re.compile(r'\(\d+\)\s*$')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        original_line = lines[i]
        
        # Пропускаем пустые строки и разделители
        if not line or line == '======':
            processed_lines.append(original_line)
            i += 1
            continue
        
        # 1. Проверяем, является ли строка отдельной сноской в скобках
        if standalone_brackets.match(line):
            removed_lines.append((i + 1, line, 'отдельная строка в скобках'))
            i += 1
            continue
        
        # 2. Проверяем, является ли строка номером сноски (только цифры)
        if footnote_number.match(line):
            # Проверяем следующую строку - если это сноска, удаляем обе
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Если следующая строка выглядит как сноска, удаляем обе
                if (standalone_brackets.match(next_line) or 
                    language_note.search(next_line) or
                    re.match(r'^[А-ЯЁ][^а-яё]*\([а-яё]+\.\)\s*$', next_line)):
                    removed_lines.append((i + 1, line, 'номер сноски'))
                    removed_lines.append((i + 2, next_line, 'текст сноски'))
                    i += 2
                    continue
            
            # Если это просто номер без сноски после, удаляем
            removed_lines.append((i + 1, line, 'номер сноски'))
            i += 1
            continue
        
        # 3. Удаляем языковые пометки в конце строк
        if language_note.search(line):
            # Удаляем пометку, но оставляем остальной текст
            cleaned_line = language_note.sub('', line).strip()
            if cleaned_line:
                processed_lines.append(cleaned_line + '\n')
                removed_lines.append((i + 1, line, 'языковая пометка удалена'))
            else:
                removed_lines.append((i + 1, line, 'строка с языковой пометкой удалена'))
            i += 1
            continue
        
        # 4. Удаляем примечания в скобках в конце строк
        if note_in_brackets.search(line):
            cleaned_line = note_in_brackets.sub('', line).strip()
            if cleaned_line:
                processed_lines.append(cleaned_line + '\n')
                removed_lines.append((i + 1, line, 'примечание удалено'))
            else:
                removed_lines.append((i + 1, line, 'строка с примечанием удалена'))
            i += 1
            continue
        
        # 5. Удаляем ссылки в скобках
        if reference_note.search(line):
            # Удаляем ссылку, но оставляем остальной текст
            cleaned_line = re.sub(r'\(см\.[^)]*\)|\(прим\.[^)]*\)', '', line).strip()
            if cleaned_line:
                processed_lines.append(cleaned_line + '\n')
                removed_lines.append((i + 1, line, 'ссылка удалена'))
            else:
                removed_lines.append((i + 1, line, 'строка со ссылкой удалена'))
            i += 1
            continue
        
        # 6. Удаляем даты в скобках в конце (только если это не часть названия)
        if date_in_brackets.search(line):
            # Проверяем, не является ли это часть названия стихотворения
            # Если строка короткая и содержит только название и дату, оставляем
            if len(line) < 50 and re.match(r'^[А-ЯЁ][^а-яё]*\(\d+\)\s*$', line):
                # Это может быть название с датой, оставляем
                processed_lines.append(original_line)
            else:
                # Удаляем дату из конца строки
                cleaned_line = date_in_brackets.sub('', line).strip()
                if cleaned_line:
                    processed_lines.append(cleaned_line + '\n')
                    removed_lines.append((i + 1, line, 'дата удалена'))
                else:
                    removed_lines.append((i + 1, line, 'строка с датой удалена'))
            i += 1
            continue
        
        # Если строка не является сноской, оставляем её
        processed_lines.append(original_line)
        i += 1
    
    # Записываем очищенный файл
    print(f"\nУдалено строк/изменений: {len(removed_lines)}")
    print(f"Исходных строк: {original_line_count}")
    print(f"Результирующих строк: {len(processed_lines)}")
    
    # Группируем по типам удалений
    by_type = {}
    for line_num, text, ftype in removed_lines:
        if ftype not in by_type:
            by_type[ftype] = []
        by_type[ftype].append((line_num, text))
    
    print("\nТипы удалений:")
    for ftype, items in sorted(by_type.items()):
        print(f"  {ftype}: {len(items)} вхождений")
        # Показываем первые примеры
        for line_num, text in items[:3]:
            print(f"    Строка {line_num}: {text[:60]}")
        if len(items) > 3:
            print(f"    ... и еще {len(items) - 3}")
    
    print(f"\nЗапись очищенного файла в {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)
    
    print("Готово!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) >= 3 else input_file
    else:
        input_file = "science/pushkin_filtered.txt"
        output_file = "science/pushkin_filtered.txt"  # Перезаписываем тот же файл
    remove_footnotes(input_file, output_file)

