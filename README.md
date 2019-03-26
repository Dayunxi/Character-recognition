# 字符识别
尝试做到多语言的手写及打印字体识别

---

## OCR

### 数据生成

由./train_data/generate_combined_char.py负责生成GB2312规定的3755个一级汉字及对应的13种字体：
[('等线', 'Deng.ttf'), ('方正舒体', 'FZSTK.TTF'), ('方正姚体', 'FZYTK.TTF'), ('仿宋', 'simfang.ttf'), ('黑体', 'simhei.ttf'), ('华文行楷', 'STXINGKA.TTF'), ('华文宋体', 'STSONG.TTF'), ('华文新魏', 'STXINWEI.TTF'), ('楷体', 'simkai.ttf'), ('隶书', 'SIMLI.TTF'), ('宋体', 'simsun.ttc'), ('微软雅黑', 'msyh.ttc'), ('幼圆', 'SIMYOU.TTF')]

为每个字生成64*64的灰度图千余张(13*72*2=1872)，字体旋转限制在30度以内，其中一半随机被腐蚀或膨胀并添加椒盐噪声

为了降低IO损耗，不将每个字单独存为一张图，而是将256*256张图片依次并入一张图，且存入磁盘前会先对所有字体图片进行shuffle，同时生成含有256*256个Label的txt文件
![train_image](./image/image_1.jpg)

### 识别

修改的LeNet, 五层卷积，四层池化
输入64X64的灰度图
conv1->pool1->conv2->pool2->conv3->pool3->conv4->conv5->pool4->fc1->fc2

暂时在1000个汉字时表现良好
学习率在[0.0015, 0.0013, 0.001, 0.0008, 0.0005]之间均收敛，其中0.0013收敛最早
![Accuracy&Loss](./image/image_2.png)