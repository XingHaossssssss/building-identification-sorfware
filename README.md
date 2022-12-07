# 地表建筑物识别

### 参考网址
https://github.com/datawhalechina/team-learning-cv/tree/master/AerialImageSegmentation

[地表建筑物识别笔记总结](https://pythontechworld.com/article/detail/7tjp3DTGENAI)
[地表建筑物识别](https://blog.csdn.net/qq_33007293/article/details/113902959)

### 文件目录

1. .idea：PyCharm自动生成的配置文件
2. __pycache__：Python解释器自动生成，存放编译的字节码
3. checkpoints：存放训练好的模型(.pth文件)，内部文件命名的格式如下：
    fcn_resnet50_15_model.pth
    fcn：模型名称
    resnet50：主干网络模型名称
    15：训练轮次，15次
    model：模型，loss文件中为loss，submit文件中为tmp
4. datasets：存放航拍地表建筑物数据集和图片
5. ico：存放程序编写时用到的图片
6. loss：存放训练过程损失函数曲线
7. submit：存放天池比赛提交结果的文件
8. 模型论文：存放本次实验使用到的模型的论文
9. data_processing.py：数据集数据处理python语言文件
10. environment.py：检测pytorch和tensorflow库配置python语言文件
11. home.py：home.ui转换的python语言文件
12. home.ui：使用Qt Designer制作程序生成的文件
13. images.qrc：Qt Designer读取程序编写使用的图片文件
14. images_rc.py：images.qrc转换的python语言文件
15. loss.py：模型训练的损失函数的python语言文件
16. main.py：模型训练预测的python语言文件
17. predict.py：模型预测的python语言文件
18. rle.py：数据集图片RLE编码和解码的python语言文件
19. test_img.py：绘制预测后图片的python语言文件
20. Tianchidataset.py：读取天池数据集的python语言文件
21. predictor.py：编写程序的python语言文件
22. tmp.csv：程序预测后生成的csv文件
23. README.md：markdown记录文件