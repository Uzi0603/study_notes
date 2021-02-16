"""
Python数据分析&数学建模 学习笔记
2020.1.13-?
山东大学 计算机学院 2019级人工智能实验班 李阳

vscode  ->  使用深色主题安装colorful comments插件提升阅读体验-.- 
anaconda3 python3.8.5 64-bit

取消注释后可以直接运行
"""

import numpy as np
import pandas as pd
import matplotlib as plt


#todo 大内容标题
#^ 小内容标题
#* 知识点
#! 注意
#~
#? 
#& 
#//


#todo Python内容回顾----------------------------------------------------
#* print('hello');print('world'); # 可以同一行显示多条语句，用分号分开即可
#* 用 \ 将一行语句分多行显示

#* x=input('enter a number:') # 将用户输入的字符以字符或字符列表的形式赋给x
# print(x)

#* 格式化输出方式1
# x=input()
# y=int(input())
# print('这是你输入的字符串:%s,这是你输入的数字:%d' % (x,y))

#^ 格式化输出方式2 .format()
# str1='数据1:{1},数据2:{0},数据3:{1}'
# print(str1.format("hello",1))
#* 创建显示样式模板时，需要使用{}和：来指定占位符
#* {[index][:[[fill]align][sign][#][width][.precision][type]]}
# index指定设置的格式作用到()中的第几个数据
# fill指定空白处填充的字符当填充字符为逗号且作用于整数或浮点数时,该整数或浮点数会以逗号分隔的形式输出,例如1000000会输出1,000,000
# align指定数据对齐方式: <左对齐 >右对齐 =右对齐,同时将符号放置在填充内容的最左侧,该选项只对数字类型有效
# sign指定有无符号数: +正数前加正号负数前加负号 -正数前不加正号负数前加负号 空格正数前加空格负数前加负号
# #对于其他进制数显示其进制前缀
# width指定数据所占的宽度
# .precison指定保留的小数位数
# type指定数据的具体类型,比如s是字符串,d是整数,e是科学计数法等

#* print(list('ly')) # ['l', 'y']
#! print(list(abc)) 报错 NameError: name 'abc' is not defined

# x=list('lynb')
#* print(x[1:3]) # 索引切片左闭右开 ['y', 'n']

#^ 常用内置函数
# print(abs(-20)) # 绝对值
# print(max(2,3,1,-5)) # 最大值，相应的min()为最小值
# print(int('123'),int(12.34),float('12.34'),str(12.34),bool(1),bool('')) # 类型转换函数

#* a=abs # 可以把函数名赋值给一个变量，这个变量就成了函数的别名
# print(a(-20))




#todo Numpy基础-------------------------------------------------------
#* data=np.random.randn(2,3) # 随机生成一个2×3数组
# print(data)

# data1=[6,7.5,8,0,1]
# data2=[[1,2,3,4],[5,6,7,8]]
#* arr1=np.array(data1) # array函数接受任意的序列型对象，生成numpy数组
#* arr2=np.array(data2) # 包含列表的列表，会生成一个二维数组
# print(arr1)
# print(arr2)
# print(arr2.shape) # 数组的属性，包括维数和每一维度的数量
#* arr2.ndim返回维数,arr2.dtype返回数据类型

#* arr3=np.zeros((3,6)) # 创建一个全0的3×6数组
#* arr4=np.ones(5) # 创建一个全1的1×5数组
#* arr5=np.empty((2,3,2)) # 创建一个未初始化数值的2×3×2数组（三维）
# print(arr3)
# print(arr4)
# print(arr5)

# arr6=np.array([1,2,3,4,5])
# print(arr6.dtype)
#* float_arr=arr6.astype(np.float64) # 数据类型转换为标准双精度浮点型
# print(float_arr.dtype)

# arr7=np.array([[1,2,3,4],[5,6,7,8]])
# print(arr7)
# print(arr7*arr7)
# print(arr7-arr7)
# print(1/arr7)
# arr8=np.array([[0,1,8,9],[4,5,7,6]])
#* print(arr8>arr7) # 同尺寸数组之间的比较会产生一个bool型数组

#^ 普通索引
# arr9=np.random.randn(3,3)
#! print(arr9[2,1]) # 效果同下,表示第三行第二列的元素
#* print(arr9[2][1]) # 二维数组第一个是行索引，第二个是列索引
#* print(arr9[1:,2:]) # 第一行到最后一行,第二列到最后一列
# print(arr9[1:][2:])
#* print(arr9[arr9>0.5]) # 返回一个一维数组,里面为bool数组中为true的元素
#* print(arr9[~(arr9>0.5)]) # ~ 取非,相应的可以用 & 限定多个条件

# arr10=np.arange(1,10) # 连续的数字1-9构成数组
#* print(arr10[-4:-2]) # 取倒数第四位到倒数第三位,即[-4,-2)
#* arr_change=arr10 # arr_change作为一个指针指向arr10,修改change则arr10也会改变
# arr_change[1:3]=[17,19]
# print(arr10)
#* arr_change1=arr10.copy() # 修改change则arr10不会改变
# arr_change1[1:3]=[2,3]
# print(arr10)

#^ 花式索引
# arr11=np.random.randn(4,5)
#* print(arr11[[2,1]]) # 按第三行和第二行的顺序输出2×2数组

#* print(arr11[[2,1],[0,1]]) # 输出arr11[2,0]和arr11[1,1]
#* print(arr11[np.ix_([2,1],[0,1])]) # 先限制第3行和第2行，再限制第1列和第2列
#* print(arr11[[2,1]][:,[0,1]]) # 同上

#* print(arr11[:,1]) # 以一维数组形式输出第二列，对行不做限制
#* print(arr11[:,[1]]) # 以原本的二维数组形式返回第二列

#^ 改变数组形状
# arr12=np.random.randn(4,5)
# print(arr12)
#* arr12.reshape(10,2) # 返回一个10行2列的视图,原数组不改变
# print(arr12.reshape(10,2))
# print(arr12)
#* print(arr12.reshape(2,-1)) # 行数确定,会自动生成列数,反之亦然
#* arr12.resize(10,2) # 改变数组形状
# print(arr12)
#* print(arr12.ravel()) # 横向降为一维,返回视图,原数组不改变
# print(arr12)
#* arr12.flatten()
#* arr12.ravel(order='F') # 纵向降为一维,返回视图,原数组不改变
# print(arr12)
#* arr12.flatten(order='F')

#! arr12.reshape(-1)[1]=2000 # 修改视图会改变原数组的值
# print(arr12)
#! arr12.ravel()[1]=1000 # 修改视图会改变原数组的值
# print(arr12)
#! arr12.flatten()[1]=2000 # 修改视图不会改变原数组的值
# print(arr12)

#* print(arr12.ndim) # 查看维数，不要用肉眼判断

# arr13=np.random.randn(4,2)
#* print(np.hstack([arr12,arr13]).shape) # 行数相同的数组可以进行横向合并
#* print(np.concatenate((arr12,arr13),axis=1).shape) # 列增加的方向(axis=1)合并，横向合并
# arr14=np.random.randn(2,5)
#* print(np.vstack([arr12,arr14]).shape) # 列数相同的数组可以进行纵向合并
#* print(np.concatenate((arr12,arr14),axis=0).shape) # 沿行增加的方向(axis=0)合并，纵向合并

#* print(np.tile(arr12,(3,2)).shape) # np.tile([数组],([行方向复制倍数],[列方向复制倍数]))

#^ 三维数组(用的不多)
# arr15=np.arange(8).reshape(2,2,2)
# print(arr15)
#! print(arr15[1][1][0]) # 第2块 第2行 第1列

#^ 数组的ufunc
#* 通用函数加减乘除只能有两个参数
# arr19=np.arange(5)
# print(arr19)
# arr20=np.array([6,3,6,3,1])
# print(arr20)
#* print(np.add(arr19,arr20)) # 加法
#* print(np.subtract(arr19,arr20)) # 减法
#* print(np.multiply(arr19,arr20)) # 乘法

# print(arr19/(arr20/100)**2)
#* print(np.divide(arr19,np.power(np.divide(arr20,100),2))) # 除法和幂
#! 书P108 数学计算所用到的函数

#* print(np.unique(arr20)) # 去除重复元素,并排列

#* print(np.in1d(arr19,arr20)) # 逐个判断19中的元素是否在20中,以数组方式返回true和false
#* print(np.intersect1d(arr19,arr20)) # 交集,并排序
#* print(np.union1d(arr19,arr20)) # 并集,并排序
#* print(np.setdiff1d(arr19,arr20)) # 差集,并排序
#* print(np.setxor1d(arr19,arr20)) # 异或集(在19或20中但不在19和20的交集中的元素)

#^ 数组的broadcasting(广播机制),对不同形状的array之间执行算术运算
# arr16=np.arange(12).reshape(3,4)
# print(arr16)
# arr17=np.arange(4)
# print(arr17)
# arr18=np.arange(3).reshape(3,-1)
# print(arr18)
#* print(arr16+arr17) # 行补齐方式,将1行复制3次形成3行4列的数组
#* print(arr16+arr18) # 列补齐方式,将1列复制4次形成3行4列数组
#* print(arr17+arr18) # 行、列均补齐
#* print(arr16==arr17) # 同样会用到广播机制补齐

#^ 通用函数中也会用到广播机制
#* print(np.greater(arr16,arr17)) # 逐个判断16中的是否比17中的大

#* print(np.greater(arr16,arr17).any()) # 只要有一个为true则返回true
#* print(np.greater(arr16,arr17).all()) # 全部为true才返回true

#* arr21=np.array([1,-1,np.nan]) # nan为形式空值
#* print(np.isnan(arr21)) # 逐个判断元素是否为空值

for i in range(5):
    print(i)