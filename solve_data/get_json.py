import json

import json

# 打开文件并读取内容    内科5000-33000_new.json是将两列删除后的
with open('E:\Word\Documents\大三下\计算机工程实训\大作业\data\儿科5-14000_new.json', 'r', encoding='utf-8') as file:
    # 使用json.load()方法解析JSON数据
    original_json = json.load(file)


converted_jsons=[]
# 将原始 JSON 数据转换为所需的格式
for item in original_json:
    # 检查 "ask" 和 "answer" 键是否都存在于字典中
    if "ask" in item and "answer" in item:
        converted_json = {
            "conversations": [
                {
                    "role": "user",
                    "content": item["ask"]
                },
                {
                    "role": "assistant",
                    "content": item["answer"]
                }
            ]
        }
        converted_jsons.append(converted_json) #<class 'list'>

# 定义新 JSON 文件的名称
new_filename = 'E:\Word\Documents\大三下\计算机工程实训\大作业\data\儿科5-14000_new_deal.json'

# 将转换后的 JSON 数据写入新文件
with open(new_filename, 'w', encoding='utf-8') as file:
    # 遍历列表中的每个 JSON 对象，并单独序列化
    for json_object in converted_jsons:
        file.write(json.dumps(json_object, ensure_ascii=False) + '\n')
# # 将转换后的 JSON 数据转换为字符串，如果需要的话
        # 序列化当前对象，并添加换行符
# converted_json_str = json.dumps(converted_json, ensure_ascii=False, indent=4)
#
# print(converted_json_st
# #     # json.dump(converted_jsons, file, ensure_ascii=False,separators=(',', ': '))
# #
# # print(f'JSON 数据已成功写入 {new_filename}')r)