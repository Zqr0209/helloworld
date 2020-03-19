import requests
import re

def requestbiliData(url):
    '''
    param url-请求的链接
    '''
    try:
        response = requests.get(url)#.post(url-请求的链接)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        return None

def parseHtmlData(htmlData):
    '''
    parma htmlData 返回的数据
    '''
    #print(htmlData)
    pattern = re.compile('.*?class="num">(.*?)</div>.*?class="info".*?'+'class="title">+(.*?)</a>.*?class="b-icon play"></i>'+'(.*?)</span>.*?class="b-icon view"></i>'+'(.*?)</span>.*?class="b-icon author"></i>'+'(.*?)</span>',re.S)
    parseData = re.findall(pattern,htmlData)

    return parseData

url = 'https://www.bilibili.com/ranking?spm_id_from=333.851.b_7072696d61727950616765546162.3' 
htmlData = requestbiliData(url) #模拟浏览器请求当当，并且返回爬取到的数据
result = parseHtmlData(htmlData) #对爬取到的数据进行解析

with open('../result/video.txt','a',encoding = 'utf-8') as f:    #存数据
    for data in result:
        print(data)
        f.write('排名：'+data[0]+'  '+'视频名称:'+data[1]+'  '+'播放量'+data[2]+'  '+'评论数'+data[3]+'  '+'UP主：'+data[4]+'  '+'\n')
