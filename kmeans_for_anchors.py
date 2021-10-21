#-------------------------------------------------------------------------------------------------#
#   kmeans铏界劧浼氬鏁版嵁闆嗕腑鐨勬杩涜鑱氱被锛屼絾鏄緢澶氭暟鎹泦鐢变簬妗嗙殑澶у皬鐩歌繎锛岃仛绫诲嚭鏉ョ殑9涓鐩稿樊涓嶅ぇ锛�
#   杩欐牱鐨勬鍙嶈�屼笉鍒╀簬妯″瀷鐨勮缁冦�傚洜涓轰笉鍚岀殑鐗瑰緛灞傞�傚悎涓嶅悓澶у皬鐨勫厛楠屾锛岃秺娴呯殑鐗瑰緛灞傞�傚悎瓒婂ぇ鐨勫厛楠屾
#   鍘熷缃戠粶鐨勫厛楠屾宸茬粡鎸夊ぇ涓皬姣斾緥鍒嗛厤濂戒簡锛屼笉杩涜鑱氱被涔熶細鏈夐潪甯稿ソ鐨勬晥鏋溿��
#-------------------------------------------------------------------------------------------------#
import glob
import xml.etree.ElementTree as ET

import numpy as np

def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])

def kmeans(box,k):
    #-------------------------------------------------------------#
    #   鍙栧嚭涓�鍏辨湁澶氬皯妗�
    #-------------------------------------------------------------#
    row = box.shape[0]
    
    #-------------------------------------------------------------#
    #   姣忎釜妗嗗悇涓偣鐨勪綅缃�
    #-------------------------------------------------------------#
    distance = np.empty((row,k))
    
    #-------------------------------------------------------------#
    #   鏈�鍚庣殑鑱氱被浣嶇疆
    #-------------------------------------------------------------#
    last_clu = np.zeros((row,))

    np.random.seed()

    #-------------------------------------------------------------#
    #   闅忔満閫�5涓綋鑱氱被涓績
    #-------------------------------------------------------------#
    cluster = box[np.random.choice(row,k,replace = False)]
    while True:
        #-------------------------------------------------------------#
        #   璁＄畻姣忎竴琛岃窛绂讳簲涓偣鐨刬ou鎯呭喌銆�
        #-------------------------------------------------------------#
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i],cluster)
        
        #-------------------------------------------------------------#
        #   鍙栧嚭鏈�灏忕偣
        #-------------------------------------------------------------#
        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break
        
        #-------------------------------------------------------------#
        #   姹傛瘡涓�涓被鐨勪腑浣嶇偣
        #-------------------------------------------------------------#
        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near

    return cluster

def load_data(path):
    data = []
    #-------------------------------------------------------------#
    #   瀵逛簬姣忎竴涓獂ml閮藉鎵綽ox
    #-------------------------------------------------------------#
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height<=0 or width<=0:
            continue
        
        #-------------------------------------------------------------#
        #   瀵逛簬姣忎竴涓洰鏍囬兘鑾峰緱瀹冪殑瀹介珮
        #-------------------------------------------------------------#
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 寰楀埌瀹介珮
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


if __name__ == '__main__':
    #-------------------------------------------------------------#
    #   杩愯璇ョ▼搴忎細璁＄畻'./VOCdevkit/VOC2007/Annotations'鐨剎ml
    #   浼氱敓鎴恲olo_anchors.txt
    #-------------------------------------------------------------#
    SIZE        = 416
    anchors_num = 9
    #-------------------------------------------------------------#
    #   杞藉叆鏁版嵁闆嗭紝鍙互浣跨敤VOC鐨剎ml
    #-------------------------------------------------------------#
    path        = r'E:/DATASET/VOCdevkit/VOC2007/Annotations'
    
    #-------------------------------------------------------------#
    #   杞藉叆鎵�鏈夌殑xml
    #   瀛樺偍鏍煎紡涓鸿浆鍖栦负姣斾緥鍚庣殑width,height
    #-------------------------------------------------------------#
    data = load_data(path)
    
    #-------------------------------------------------------------#
    #   浣跨敤k鑱氱被绠楁硶
    #-------------------------------------------------------------#
    out = kmeans(data,anchors_num)
    out = out[np.argsort(out[:,0])]
    print('acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print(out*SIZE)
    data = out*SIZE
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()
