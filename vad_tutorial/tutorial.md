# VAD tutorial

### 任务定义
语音端点检测(voice activity detection,VAD)，就是在有噪声的环境中区分出语音部分和非语音部分,并判断出语音部分的起点和终点。

### 模型目录结构
下面**不是**完整的文件目录结构，而是与流程相关的主要文件目录
```
/slwork/users/yfc07/vad

vad                                     顶层目录
└── tutorial                            项目目录
    ├── run.sh                          集群任务提交脚本
    ├── data                            数据目录(特征，标注等)
    ├── exp                             模型训练输出目录
    │   ├── exp1                        各个模型对应的目录
    │   └── exp2
    ├── log                             日志文件目录
    ├── src                             训练脚本等代码目录
    │   ├── exp1                        各个模型对应的训练代码
    │   └── exp2
    └── testdir                         测试目录
        ├── tools                       后处理及打分脚本
        ├── exp1                        各个模型对应的测试目录
        └── exp2
```

### tutorial数据集
该例子中使用的数据集是aurora4，训练集是其中的 **train_si84_multi** ，是一个包含了noise和clean音频的数据集，测试集是 **test_eval92**，包含了A，B，C，D四个子集合，分别是clean，noise，clean+信道失真，noise+信道失真

### 基本流程(以dnn_ce为例)
1. 特征提取，准备标注

    - 目前vad训练使用的特征主要是fbank，提取特征的工具有htk或是kaldi，该tutorial使用的是kaldi格式的24维filter bank特征，add-delta=2，帧扩展为前后各5帧
    - 数据的标注是通过force alignment得到的`tutorial/data/ali/train.ali`

2. 模型训练

    训练部分使用的神经网络工具是pytorch_0.4.1，脚本在`tutorial/src/dnn_ce`目录中，主要包括了：
    - dataloader.py
    - models.py
    - train.py

    分别用作数据读入，模型建立和模型训练，执行train.py脚本之后，训练得到模型文件以及log输出到 **exp** 下对应的模型目录内，具体提交命令见后

3. 结果后处理

    在vad的任务中，后处理是很有必要的，因为模型的原始输出往往存在着比较多的碎片，即sil和speech之间的错误转换，由于语音信号是有很强的连续性的，所以通过平滑原始的输出结果可以获得更加准确的区分结果，后处理的脚本是`tutorial/testdir/tools/post_process.py`，输入输出都是mlf格式的结果文件，作用就是设置窗长来对结果进行平滑，去除不合理的碎片，调用格式如下

    ```
    python post_process.py in_mlf_file out_mlf_file filter_size
    ```

4. 测试

    测试的脚本是`tutorial/src/dnn_ce/test.py`，即相同的提取测试数据的特征过模型，用后验概率来进行判别，最后会在 **testdir** 里的对应模型目录下输出的mlf格式的文件，具体调用命令见后

### 任务提交以及执行流程
1. 环境要求：python3，numpy，kaldi，pytorch_0.4.1
2. 实验目录：`/slwork/users/yfc07/vad/tutorial`
3. 训练任务提交格式：
    - `qsub -cwd -S /bin/bash -q gpu.q -l hostname=turing -o log/dnn_ce.log -j y run.sh`
    - run.sh脚本中首先加载kaldi环境，然后调用`src/dnn/train.sh`，具体参数设置：
    - `python src/dnn_ce/train.py  --lr 5e-5 --epochs 10 --batch_size 256 --lr_decay --cuda --exp dnn_ce --log_batch_interval 2000`
    - 小集群机器配置有差异，最好提交到turing上
4. 测试任务提交格式与训练类似，脚本调用格式为
    - `python src/dnn_ce/test.py --exp dnn_ce`
5. 后处理及计算acc，vacc
    - 计算结果的脚本是：`tutorial/testdir/tools/criterion.py`
    - 调用格式为`python criterion.py result_mlf_file reference_mlf_file`，即将网络输出经过后处理的mlf格式的结果同标注结果对比，计算指标
    - 测试集的标注：`tutorial/data/test_eval92/mlf`
6. lstm模型的调用脚本为`run_lstm.sh`



