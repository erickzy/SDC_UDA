# SDC ND P1- lane line find
## 车道线视频地址
[SDC_white](http://oqakc4551.bkt.clouddn.com/solidWhiteRighttest2.mp4)

[SDC_Yellow](http://oqakc4551.bkt.clouddn.com/solidYellowLefttest.mp4)

## 相关代码请参考
`SDC_ND_P1_lane_line_delivery.ipynb`
相关内容为project提交内容

`video_lane_line.py` & `video_lane_line_challenge.py`
中为针对普通的和challenge的不同标定的代码。
challenge目前状态还是有问题，特别是在光线及路面变化的过程中对于车道线的识别十分不好。

车道选取的case难点在于对车道线在不同工况下识别的分析和判断。

鉴于时间紧张先提交一版，后续计划修改的点包括：
1. 标定参数统一标识，函数中内容采用全局变量替换
2. 车道线识别应更具有广泛性，需要重新标定
3. 如何才能出现稳定的车道线需要向TA请教，目前车道线在部分路面会出现断断续续或者斜率不稳定的状态。




