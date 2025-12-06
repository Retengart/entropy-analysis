#!/usr/bin/env python3
"""
Скрипт для удаления примечаний редактора из отфильтрованного файла.
Удаляет строки, начинающиеся с типичных фраз примечаний редактора.
"""

import re


def is_editor_note(line: str) -> bool:
    """
    Проверяет, является ли строка примечанием редактора.
    """
    line_stripped = line.strip()
    if not line_stripped:
        return False
    
    # Паттерны начала примечаний редактора
    note_patterns = [
        r'^Печатается по',
        r'^Датируется',
        r'^Впервые опубликовано',
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
        r'^В \d+ году Лермонтов',
        r'^По словам',
        r'^Анна Григорьевна',
        r'^В Гродненском',
        r'^Козлов –',
        r'^Рядом с текстом',
        r'^Оно действительно',
        r'^Это показание',
        r'^В \d+ году',
        r'^Стих \d+',
        r'^Строки \d+',
        r'^В «Стихотворениях»',
        r'^В автографе',
        r'^Какая.*имеется в виду',
        r'^листок\)\.',
        r'^оставляем первоначальное',
        r'^Следующие.*стихотворений',
        r'^В рукописи',
        r'^В стихе \d+',
        r'^чтение стиха',
    ]
    
    for pattern in note_patterns:
        if re.match(pattern, line_stripped, re.IGNORECASE):
            return True
    
    # Также проверяем строки, которые содержат типичные элементы примечаний
    if re.search(r'ИРЛИ|ГИМ|ГПБ|л\. \d+|стр\. \d+|т\. \d+|№ \d+|оп\. \d+|ф\. \d+', line_stripped):
        # Но не удаляем, если это может быть часть стиха (например, содержит рифму)
        # Проверяем, не похоже ли это на стих (короткие строки, рифма)
        if len(line_stripped) < 100 and not re.search(r'[.!?]$', line_stripped):
            # Может быть стих, пропускаем
            return False
        # Длинные строки с архивными ссылками - это примечания
        return True
    
    # Проверяем строки с типичными фразами примечаний в середине или конце
    if re.search(r'Черновой автограф|позднейшая приписка|см\. примечание|не установлено|исправлена по автографу|исправляется по|воспроизводятся:|записках»|запискам»', line_stripped, re.IGNORECASE):
        return True
    
    # Проверяем строки, которые являются продолжением примечаний (содержат ссылки на издания)
    if re.search(r'Библиогр\.|Отеч\. записках|Стихотворениях.*Лермонтова|Соч\. под ред\.|Русск\.|Лит\.', line_stripped):
        # Но не удаляем короткие строки, которые могут быть частью стиха
        if len(line_stripped) > 80:  # Длинные строки с ссылками - это примечания
            return True
    
    # Строки, начинающиеся с цифр и содержащие ссылки - это продолжение примечаний
    if re.match(r'^\d+', line_stripped) and re.search(r'стр\.|т\.|№|Соч\.|под ред\.|отд\.', line_stripped):
        return True
    
    # Строки, начинающиеся со скобок и содержащие архивные ссылки или фразы примечаний
    if re.match(r'^\(', line_stripped) and (re.search(r'ИРЛИ|ГИМ|ГПБ|заголовком|копия|сделана', line_stripped) or len(line_stripped) > 60):
        return True
    
    # Строки с фразами типа "Впервые были опубликованы"
    if re.search(r'Впервые.*опубликован|опубликованы.*только|были опубликованы', line_stripped, re.IGNORECASE):
        return True
    
    # Строки с фразами типа "Е. А. Сушкова в своих «Записках»"
    if re.search(r'в своих.*Записках|поставила дату|после стихотворения', line_stripped, re.IGNORECASE):
        return True
    
    # Строки, начинающиеся с маленькой буквы и содержащие ссылки (продолжение примечаний)
    if line_stripped[0].islower() and re.search(r'стр\.|т\.|№|Соч\.|под ред\.|стихотворению|датировку|монастырь', line_stripped) and len(line_stripped) > 40:
        return True
    
    return False


def remove_editor_notes(input_file: str, output_file: str):
    """Удаляет примечания редактора из файла."""
    print(f"Чтение файла {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Разделяем на стихи
    poems = content.split('======')
    
    cleaned_poems = []
    total_removed = 0
    
    for poem in poems:
        lines = poem.strip().split('\n')
        cleaned_lines = []
        in_note_block = False
        
        prev_line_was_note = False
        for line in lines:
            line_stripped = line.strip()
            
            # Пропускаем пустые строки
            if not line_stripped:
                # Если мы в блоке примечаний, продолжаем пропускать
                if in_note_block:
                    prev_line_was_note = True
                    continue
                # Иначе добавляем пустую строку для форматирования
                cleaned_lines.append('')
                prev_line_was_note = False
                continue
            
            # Проверяем, является ли строка примечанием
            if is_editor_note(line_stripped):
                in_note_block = True
                prev_line_was_note = True
                total_removed += 1
                continue
            
            # Если мы в блоке примечаний или предыдущая строка была примечанием,
            # проверяем, не является ли текущая строка продолжением
            if in_note_block or prev_line_was_note:
                is_continuation = False
                
                # Строки, начинающиеся с цифр, скобок - часто продолжение примечаний
                if re.match(r'^\d+|^\(|^«', line_stripped):
                    is_continuation = True
                
                # Строки, начинающиеся с маленькой буквы после примечания - продолжение
                if prev_line_was_note and line_stripped[0].islower() and len(line_stripped) > 30:
                    is_continuation = True
                
                # Строки с архивными ссылками или ссылками на издания
                if re.search(r'ИРЛИ|ГИМ|ГПБ|л\. \d+|стр\. \d+|т\. \d+|№ \d+|оп\.|ф\.|Соч\.|под ред\.|записках|запискам|Библиогр\.|Отеч\.|Русск\.|Лит\.|заголовком|стихотворению|стихотворения|датировку|монастырь|Стихотворениях.*Лермонтова', line_stripped):
                    is_continuation = True
                
                # Длинные строки (более 80 символов) с признаками примечаний
                if len(line_stripped) > 80 and re.search(r'Соч\.|под ред\.|записках|запискам|стр\.|т\.|№', line_stripped):
                    is_continuation = True
                
                # Короткие строки, заканчивающиеся точкой или скобкой после примечания
                if prev_line_was_note and len(line_stripped) < 80 and re.search(r'[\.\)»]$', line_stripped):
                    is_continuation = True
                
                if is_continuation:
                    # Продолжение примечания
                    in_note_block = True
                    prev_line_was_note = True
                    total_removed += 1
                    continue
                else:
                    # Это начало нового текста стиха
                    in_note_block = False
                    prev_line_was_note = False
            
            cleaned_lines.append(line_stripped)
            prev_line_was_note = False
        
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
    remove_editor_notes(input_file, output_file)

