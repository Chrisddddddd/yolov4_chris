import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    '''
    Recall鍜孭recision涓嶅儚AP鏄竴涓潰绉殑姒傚康锛屽湪闂ㄩ檺鍊间笉鍚屾椂锛岀綉缁滅殑Recall鍜孭recision鍊兼槸涓嶅悓鐨勩��
    map璁＄畻缁撴灉涓殑Recall鍜孭recision浠ｈ〃鐨勬槸褰撻娴嬫椂锛岄棬闄愮疆淇″害涓�0.5鏃讹紝鎵�瀵瑰簲鐨凴ecall鍜孭recision鍊笺��

    姝ゅ鑾峰緱鐨�./map_out/detection-results/閲岄潰鐨則xt鐨勬鐨勬暟閲忎細姣旂洿鎺redict澶氫竴浜涳紝杩欐槸鍥犱负杩欓噷鐨勯棬闄愪綆锛�
    鐩殑鏄负浜嗚绠椾笉鍚岄棬闄愭潯浠朵笅鐨凴ecall鍜孭recision鍊硷紝浠庤�屽疄鐜癿ap鐨勮绠椼��
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode鐢ㄤ簬鎸囧畾璇ユ枃浠惰繍琛屾椂璁＄畻鐨勫唴瀹�
    #   map_mode涓�0浠ｈ〃鏁翠釜map璁＄畻娴佺▼锛屽寘鎷幏寰楅娴嬬粨鏋溿�佽幏寰楃湡瀹炴銆佽绠梀OC_map銆�
    #   map_mode涓�1浠ｈ〃浠呬粎鑾峰緱棰勬祴缁撴灉銆�
    #   map_mode涓�2浠ｈ〃浠呬粎鑾峰緱鐪熷疄妗嗐��
    #   map_mode涓�3浠ｈ〃浠呬粎璁＄畻VOC_map銆�
    #   map_mode涓�4浠ｈ〃鍒╃敤COCO宸ュ叿绠辫绠楀綋鍓嶆暟鎹泦鐨�0.50:0.95map銆傞渶瑕佽幏寰楅娴嬬粨鏋溿�佽幏寰楃湡瀹炴鍚庡苟瀹夎pycocotools鎵嶈
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #-------------------------------------------------------#
    #   姝ゅ鐨刢lasses_path鐢ㄤ簬鎸囧畾闇�瑕佹祴閲廣OC_map鐨勭被鍒�
    #   涓�鑸儏鍐典笅涓庤缁冨拰棰勬祴鎵�鐢ㄧ殑classes_path涓�鑷村嵆鍙�
    #-------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #-------------------------------------------------------#
    #   MINOVERLAP鐢ㄤ簬鎸囧畾鎯宠鑾峰緱鐨刴AP0.x
    #   姣斿璁＄畻mAP0.75锛屽彲浠ヨ瀹歁INOVERLAP = 0.75銆�
    #-------------------------------------------------------#
    MINOVERLAP      = 0.5
    #-------------------------------------------------------#
    #   map_vis鐢ㄤ簬鎸囧畾鏄惁寮�鍚疺OC_map璁＄畻鐨勫彲瑙嗗寲
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   鎸囧悜VOC鏁版嵁闆嗘墍鍦ㄧ殑鏂囦欢澶�
    #   榛樿鎸囧悜鏍圭洰褰曚笅鐨刅OC鏁版嵁闆�
    #-------------------------------------------------------#
    VOCdevkit_path  = 'E:/DATASET/VOCdevkit'
    #-------------------------------------------------------#
    #   缁撴灉杈撳嚭鐨勬枃浠跺す锛岄粯璁や负map_out
    #-------------------------------------------------------#
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = 0.001, nms_iou = 0.5)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/images/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
