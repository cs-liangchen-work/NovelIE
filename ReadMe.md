Code for emnlp2023 paper: Novel Slot Detection With an Incremental Setting

核心代码在train中：
- A_data.py 是数据处理格式
- A_model.py 是模型架构，调用了matcher.py中定义的函数
- A_train.py 是训练+测试代码，train.sh会运行该函数
- A_convert.py 是模型保存和加载时因调用huggingface代码不同，有可能key值发生变化，提供的一个转换代码。
