#!/usr/bin/env python3
"""
Скрипт для исправления ошибок OCR в отфильтрованном файле pushkin_filtered.txt
"""

import re


def fix_ocr_errors(input_file: str, output_file: str):
    """Исправляет ошибки OCR в файле."""
    print(f"Чтение файла {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    fixes_made = []
    
    # 1. Исправляем немецкую кавычку „ на обычную "
    count1 = content.count('„')
    if count1 > 0:
        content = content.replace('„', '"')
        fixes_made.append(f"Немецкая кавычка „ → \" ({count1} раз)")
    
    # 2. Исправляем угловые скобки вокруг части слова (например, М<артынов> → Мартынов)
    # Паттерны: буква<буквы>буква, буква<буквы> пробел/знак препинания
    def fix_partial_word_brackets(match):
        before = match.group(1)
        inside = match.group(2)
        after = match.group(3) if match.lastindex >= 3 else ''
        # Если внутри только буквы (не редакторская пометка), убираем скобки
        if re.match(r'^[а-яёА-ЯЁa-zA-Z]+$', inside):
            return before + inside + after
        return match.group(0)  # Оставляем как есть
    
    # Паттерн 1: буква<буквы>пробел/знак препинания/конец строки
    pattern1a = re.compile(r'(\w)<([а-яёА-ЯЁa-zA-Z]+)>([\s\W]|$)')
    count2a = len(pattern1a.findall(content))
    if count2a > 0:
        content = pattern1a.sub(fix_partial_word_brackets, content)
        fixes_made.append(f"Угловые скобки вокруг части слова ({count2a} раз)")
    
    # Паттерн 2: буква<буквы>буква
    pattern1b = re.compile(r'(\w)<([а-яёА-ЯЁa-zA-Z]+)>(\w)')
    count2b = len(pattern1b.findall(content))
    if count2b > 0:
        content = pattern1b.sub(fix_partial_word_brackets, content)
        fixes_made.append(f"Угловые скобки вокруг части слова в середине ({count2b} раз)")
    
    # 3. Исправляем пустые угловые скобки с тире: <– > → –
    count3 = content.count('<– >')
    if count3 > 0:
        content = content.replace('<– >', '–')
        fixes_made.append(f"Пустые угловые скобки с тире <– > → – ({count3} раз)")
    
    # 4. Исправляем угловые скобки с многоточием: <……> → [...]
    pattern2 = re.compile(r'<\.{2,}>')
    matches2 = pattern2.findall(content)
    count4 = len(matches2)
    if count4 > 0:
        content = pattern2.sub('[...]', content)
        fixes_made.append(f"Угловые скобки с многоточием <……> → [...] ({count4} раз)")
    
    # 4a. Исправляем двойные кавычки подряд: "" → " (но не трогаем правильные парные кавычки)
    # Ищем паттерн: "текст"текст - это ошибка, должно быть "текст" текст
    pattern4a = re.compile(r'"([^"]*)"([^"\s])')
    count4a = len(pattern4a.findall(content))
    if count4a > 0:
        def fix_double_quotes(match):
            text = match.group(1)
            next_char = match.group(2)
            return f'"{text}" {next_char}'
        content = pattern4a.sub(fix_double_quotes, content)
        fixes_made.append(f"Двойные кавычки подряд исправлены ({count4a} раз)")
    
    # 4b. Исправляем случаи типа: "текст"текст (без пробела между кавычками)
    pattern4b = re.compile(r'"([^"]+)"([а-яёА-ЯЁa-zA-Z])')
    count4b = len(pattern4b.findall(content))
    if count4b > 0:
        def fix_quotes_no_space(match):
            text = match.group(1)
            next_char = match.group(2)
            return f'"{text}" {next_char}'
        content = pattern4b.sub(fix_quotes_no_space, content)
        fixes_made.append(f"Кавычки без пробела после исправлены ({count4b} раз)")
    
    # 5. Исправляем угловые скобки с пробелами и тире: < – > → –
    count5 = content.count('< – >')
    if count5 > 0:
        content = content.replace('< – >', '–')
        fixes_made.append(f"Угловые скобки с пробелами < – > → – ({count5} раз)")
    
    # 6. Исправляем одиночные угловые скобки вокруг одного символа или коротких слов (кроме редакторских пометок)
    # Например: <я>, <из> в середине текста должно быть просто "я", "из"
    # Но оставляем редакторские пометки типа <На Карамзина>, <Из письма>
    def fix_single_char_brackets(match):
        full_match = match.group(0)
        inside = match.group(1).strip()
        # Если это редакторская пометка (начинается с заглавной буквы, содержит точку/запятую/вопрос/многоточие), оставляем
        if (re.match(r'^[А-ЯЁ]', inside) or 
            '.' in inside or ',' in inside or '?' in inside or 
            '…' in inside or '–' in inside or '—' in inside or
            len(inside) > 10):  # Длинные пометки оставляем
            return full_match
        # Если это короткое слово (1-3 буквы) в угловых скобках, убираем скобки
        if len(inside) <= 3 and re.match(r'^[а-яёА-ЯЁa-zA-Z]+$', inside):
            return inside
        return full_match
    
    pattern3 = re.compile(r'<([^>]+)>')
    # Подсчитываем только те, которые нужно исправить
    single_char_fixes = 0
    for match in pattern3.finditer(content):
        inside = match.group(1).strip()
        if (len(inside) <= 3 and 
            re.match(r'^[а-яёА-ЯЁa-zA-Z]+$', inside) and
            not re.match(r'^[А-ЯЁ]', inside) and
            '.' not in inside and ',' not in inside and '?' not in inside):
            single_char_fixes += 1
    
    if single_char_fixes > 0:
        content = pattern3.sub(fix_single_char_brackets, content)
        fixes_made.append(f"Короткие слова в угловых скобках (кроме редакторских пометок) ({single_char_fixes} раз)")
    
    # Записываем исправленный файл
    if content != original_content:
        print(f"\nВнесены исправления:")
        for fix in fixes_made:
            print(f"  - {fix}")
        
        print(f"\nЗапись исправленного файла в {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Готово!")
    else:
        print("Изменений не требуется.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) >= 3 else input_file
    else:
        input_file = "science/pushkin_filtered.txt"
        output_file = "science/pushkin_filtered.txt"  # Перезаписываем тот же файл
    fix_ocr_errors(input_file, output_file)

