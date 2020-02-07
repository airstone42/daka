# Daka

一个用于自动登陆、每日打卡的简单工具。使用 CNN 训练模型，自动识别登陆页的验证码。

![](https://cas.shmtu.edu.cn/cas/captcha)

## Dependencies

### Java

- [cage](https://github.com/akiraly/cage)

### Python

- opencv2
- keras
- numpy
- pandas
- yaml
- requests
- Beautiful Soup

## Deployment

```cage``` 目录下为 Java 程序，使用 cage 库生成验证码，与目标网站登陆页所采用的验证码方案一致。

```train.py``` 将生成的验证码图片作为数据集，使用 CNN 训练模型。

由于目标网站登陆页的验证码由两位数字构成，这里直接根据宽度将目标图片裁剪为左、右两边，分块识别。左、右两个模型，目标类别数目为 10 ，准确率大约为 99% 。

```data``` 目录下为训练所得的模型 ```left.h5``` 和 ```right.h5``` ，以及计算所得数据集的均值、标准差 ```numbers.csv``` ，用于数据预处理。

参照 ```example.yml``` 创建 ```config.yml``` ，内容包括登陆页所需的用户名以及密码。运行 ```main.py``` 进行自动登陆及打卡。

配置文件中的其他字段为每日打卡页面所需的选项，根据个人实际情况填写。

若无需使用登陆、打卡功能，仅验证验证码结果，可以只使用 ```main.py``` 中的 ```recognize()``` 进行验证。

## References

[目标网站的原型中验证码生成部分](https://github.com/kawhii/sso/blob/master/sso-support/sso-support-captcha/src/main/java/com/carl/sso/support/captcha/imp/cage/CageStringCaptchaWriter.java)

[Deep Computer Vision Using Convolutional Neural Networks](https://github.com/ageron/handson-ml2/blob/master/14_deep_computer_vision_with_cnns.ipynb)
