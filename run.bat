@echo off
echo Training started at %date% %time% > training_log.log
call D:\huangxiaoke\Anaconda3\Scripts\activate gluon
call python neural_style.py -ic .\img\BNU.jpg -is .\img\style_0.jpg -o out_1.png >> training_log.log
call python neural_style.py -ic .\img\BNU.jpg -is .\img\style_1.jpg -o out_2.png >> training_log.log
call python neural_style.py -ic .\img\Forest.jpg -is .\img\style_0.jpg -o out_3.png >> training_log.log
call python neural_style.py -ic .\img\Forest.jpg -is .\img\style_1.jpg -o out_4.png >> training_log.log
pause