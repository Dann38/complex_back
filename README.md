# Бинаризация документов со сложным фоном

Решается задача по удалению сложного фона документа, для того чтобы программа [tesseract](https://github.com/tesseract-ocr/tesseract)
могла корректно считать текст.

*Используемые библиотеки находятся в файле requirements.txt*

---
Все методы реализованны в виде функций и хранятся в папке ```lib``` разбитые по функции

---
Файл обработки называется ```img_processing```

---
Графическое отображение результата анализа можно получить запустив файл 
```gui_read_img.py имя_файла```

---
### Тестирование метода
Для тестирования необходимо создать папку и туда поместить изучаемые изображения [и текст результата для сверки (назвать одинаково)]
для программы нужно также указать папку для сохранения результата (!В ПАПКЕ РЕЗУЛЬТАТОВ ВСЕ СОДЕРЖИМОЕ ДО УДАЛИТСЯ!)

После запускаем
```script.py -i папка_с_исходниками -o папка_с_результатом -ow```



