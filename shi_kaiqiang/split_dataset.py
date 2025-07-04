#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import random
from pathlib import Path

# 定义数据集路径
DATASET_PATH = Path('face_database')
TRAIN_PATH = DATASET_PATH / 'train'
VAL_PATH = DATASET_PATH / 'val'
TEST_PATH = DATASET_PATH / 'test'

# 创建验证集和测试集目录
VAL_PATH.mkdir(exist_ok=True)
TEST_PATH.mkdir(exist_ok=True)

# 设置随机种子以确保结果可重复
random.seed(42)

# 数据集划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 打印开始信息
print(f"开始将数据集分割为训练集 ({TRAIN_RATIO*100}%)、验证集 ({VAL_RATIO*100}%) 和测试集 ({TEST_RATIO*100}%)")

# 获取所有人名目录
persons = [d for d in TRAIN_PATH.iterdir() if d.is_dir()]
print(f"找到 {len(persons)} 个人的图像文件夹")

# 为每个人创建对应的验证集和测试集目录
for person_dir in persons:
    person_name = person_dir.name
    
    # 创建对应的验证集和测试集目录
    (VAL_PATH / person_name).mkdir(exist_ok=True)
    (TEST_PATH / person_name).mkdir(exist_ok=True)
    
    # 获取该人的所有图像
    images = list(person_dir.glob('*.jpg'))
    random.shuffle(images)  # 随机打乱图像顺序
    
    # 计算各集合中应有的图像数量
    total_images = len(images)
    val_count = int(total_images * VAL_RATIO)
    test_count = int(total_images * TEST_RATIO)
    train_count = total_images - val_count - test_count
    
    # 划分数据集
    val_images = images[:val_count]
    test_images = images[val_count:val_count + test_count]
    # 训练集图像保留在原位置
    
    print(f"处理 {person_name}: 总计 {total_images} 张图像")
    print(f"  - 训练集: {train_count} 张图像")
    print(f"  - 验证集: {val_count} 张图像")
    print(f"  - 测试集: {test_count} 张图像")
    
    # 移动验证集图像
    for img in val_images:
        dest = VAL_PATH / person_name / img.name
        shutil.copy2(img, dest)
        print(f"  移动到验证集: {img.name}")
    
    # 移动测试集图像
    for img in test_images:
        dest = TEST_PATH / person_name / img.name
        shutil.copy2(img, dest)
        print(f"  移动到测试集: {img.name}")
    
    # 从训练集中删除被移动到验证集和测试集的图像
    for img in val_images + test_images:
        os.remove(img)
        print(f"  从训练集删除: {img.name}")

print("数据集划分完成！") 