import jieba

text='我本科毕业于渤海大学计算机学院，现在渤海大学城市学院当老师。'
word_list=jieba.cut(text)
print(list(word_list))
#-->:['我', '本科毕业', '于', '渤海', '大学', '计算机', '学院', '，', '现在', '渤海', '大学', '城市', '学院', '当', '老师', '。']

word_list=jieba.cut(text, cut_all= False)
print('精确模式分词：'+"/".join(word_list))

word_list=jieba.cut(text, cut_all= True)
print('全模式分词：'+"/".join(word_list))

word_list=jieba.cut_for_search(text)
print('搜素引擎模式分词：'+"/".join(word_list))

word_list=jieba.lcut(text)
print("自动转换成列表：%s"%word_list)

#装载自定义字典
jieba.load_userdict('mydict.txt')
word_list=jieba.lcut(text)
print("自动转换成列表：%s"%word_list)

