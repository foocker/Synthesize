#! /bin/usr/env python3
# -*- coding:utf-8 -*-
import time

from PIL import Image, ImageDraw
from pyzbar.pyzbar import decode

import requests
from bs4 import BeautifulSoup

import sys

from proxy import Proxies

from barcode_detect import barcode_rotated

# https://zh.wikipedia.org/wiki/%E6%AC%A7%E6%B4%B2%E5%95%86%E5%93%81%E7%BC%96%E7%A0%81

def BarcodeDict(file):
    img = Image.open(file).convert('RGB')
    barcodes = decode(img)
    if len(barcodes):
        bc = barcodes[0]
        return {'CodesType':bc.type, 'codenumber':bc.data.decode('gbk')}
    print('no barcode')
    return


def get_cookie():
    url = "http://search.anccnet.com/writeSession.aspx?responseResult=check_ok"
    response = requests.get(url)
    # print(response.cookies.get_dict())
    return response.cookies.get_dict()


def BarcodeInfo(bar_number):
    url = "http://search.anccnet.com/searchResult2.aspx?keyword={}".format(bar_number)

    header = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Host": "search.anccnet.com",
        "Referer": "http://www.gds.org.cn/"
    }
    response = requests.get(url, headers=header, cookies=get_cookie(), proxies=Proxies().get_proxies())
    response.encoding = "gbk"

    soup = BeautifulSoup(response.text, 'lxml')
    dt_list = soup.select(r'div[class="result"] dl dt')
    dd_list = soup.select(r'div[class="result"] dl dd')
    item = {}
    for i in range(len(dt_list)):
        item[dt_list[i].get_text().strip().replace("：", "")] = dd_list[i].get_text().strip()
    return item


if __name__ == "__main__":
    import os
    # print(sys.getdefaultencoding())
    # bar_num_list = [6901028193498, 6901028221504, 6901028188227, 6901028182652,6901028072540, 222]
    # for b in bar_num_list:
    #     print(BarcodeInfo(b))

    # test barcode_detect.py
    st = time.time()
    count_n = 0
    for f in os.listdir(r'../images'):
        if os.path.isfile(os.path.join('../images', f)) and f.endswith('jpg'):
            barinfo = barcode_rotated(os.path.join('../images', f))
            if barinfo is None:
                continue
            dd = BarcodeInfo(barinfo['codenumber'])
            count_n += 1
            print(f, barinfo['codenumber'], dd)
    avg_time = (time.time() - st)/count_n
    print('one cig cost time is {}'.format(avg_time))



