import requests, json, csv, threading, queue, time






# info, nicheInfo, contactTime, costCount, levelVal, level, timeOfDay, putIntoCostLevel, putIntoTimeLevel, cognitionCillVal = []

def get_info(i):
    url = "http://hobby.lkszj.info/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.33",
        "Content-Type": "application/json",
    }
    data = json.dumps({"searchStr":"", "pageSize":"15", "pageNum":str(i),"order":""})
    res = requests.post(url, data=data, headers=headers)
    res.encoding = "utf-8"
    res = res.json()["list"]
    print("第" + str(i) + "页")
    for j in range(14):
        temp = []
        # 爱好
        temp.append(res[j]["info"])
        # 小众程度
        temp.append(res[j]["nicheInfo"])
        # 入坑时间
        temp.append(res[j]["contactTime"])
        # 总花销
        temp.append(res[j]["costCount"])
        # 领域内程度描述
        temp.append(res[j]["level"])
        # 领域内程度等级
        temp.append(res[j]["levelVal"])
        # 每日时长
        temp.append(res[j]["timeOfDay"])
        # 金钱投入门槛
        temp.append(res[j]["putIntoCostLevel"])
        # 时间投入门槛
        temp.append(res[j]["putIntoTimeLevel"])
        # 社会认知程度描述
        temp.append(res[j]["cognitionCill"])
        # 社会认知程度等级
        temp.append(res[j]["cognitionCillVal"])
    info.put(temp)
    return 0

# save


if __name__ == "__main__":
    start_time = time.time()
    url = "http://hobby.lkszj.info/search"
    data = json.dumps({"searchStr":"", "pageSize":"15", "pageNum":"1","order":""})
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.33",
        "Content-Type": "application/json",
        }
    res = requests.post(url, data=data, headers=headers).json()
    count = res['count']
    info = queue.Queue()
    task = []
    for i in range(1, int(count/15) + 1):
        task.append( threading.Thread(target = get_info, args = (i,) ) )
        task[-1].start()

    while True:
        live = len(task)
        print("检测线程")
        for i in range(len(task)):
            if not task[i].is_alive():
                live -= 1
            elif live == 1:
                print(task[i])
        run_time = time.time() - start_time
        time.sleep(0.2)
        if not live or run_time > int(count/15) / 10:
            data = []
            while not info.empty():
                data.append(info.get())
            with open("hobby.csv", "w", newline='', encoding = "utf-8-sig") as csvfile: 
                writer = csv.writer(csvfile)
                #先写入 columns_name
                writer.writerow(["爱好", "小众程度", "入坑时间", "总花销", "领域内程度等级", "领域内程度描述", "每日时长", "金钱投入门槛", "时间投入门槛", "社会认知程度描述", "社会认知程度等级"])
                #写入多行用 writerows
                writer.writerows(data)
                break