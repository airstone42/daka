# Daka

一个用于自动登陆、每日打卡的简单工具。
使用 CNN 训练模型，自动识别登陆页的验证码。


## Dependencies

### Java
- cage

### Python
- opencv2
- keras
- numpy
- pandas
- yaml
- requests
- beautifulsoup


## Deployment

```cage``` 目录下是 Java 程序，使用 cage 生成验证码，与目标网站登陆页所使用的验证码方案一致。 

```train.py``` 利用生成的验证码图片为数据集，使用 CNN 训练模型。

由于目标网站登陆页的验证码由两位数字构成，这里直接采取整体识别，目标类别数目为 100 ，准确率大约为 99% 。

参照 ```example.yml``` 创建 ```config.yml``` ，内容包括登陆页所需的用户名以及密码。运行 ```main.py``` 进行自动登陆及打卡。

配置文件中的其他字段为每日打卡页面所需的选项，根据个人实际情况填写。
