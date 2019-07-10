import requests  #http lib
from bs4 import BeautifulSoup  #climb
import os,sys
import traceback
def downloadImgs(url, filename):
    if os.path.exists(filename):
        return
    try: #stream模式处理较大的文件
        request = requests.get(url=url, stream=True, timeout=60) 
        #如果get请求有异常，则会抛出，否则为None
        request.raise_for_status() 
        with open(filename,'wb') as f:
            #迭代读取内容，每次1024个字节
            for chunk in request.iter_content(chunk_size=1024):  
                if chunk:
                    f.write(chunk)
                    f.flush()
        return filename
    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        return KeyboardInterrupt
    except Exception:
        traceback.print_exc()
        if os.path.exists(filename):
            os.remove(filename)

begin = int(sys.argv[1])
end = int(sys.argv[2])
for i in range(begin, end):
    url = 'http://konachan.net/post?page=%d&tags=' % i  #动漫头像爬取地址
    html = requests.get(url).text  #获取网页源代码信息
    soup = BeautifulSoup(html, 'html.parser') #解析html字段
    for image in soup.find_all('img', class_='preview'): #html字段中的preview类里面寻找图片
        image_url = image['src']
        filename = os.path.join('image_raw', image_url.split('/')[-1])
        downloadImgs(image_url, filename)
    print('download %d page' %i)