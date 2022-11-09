from hashlib import new
from torch import true_divide
from ana.model import FunctionBase, ModuleFunction
from yolo_9000.darknet.python.darknet import yoloVIDVIP
from eq2cm import eq_to_pers
import cv2
import numpy as np
import pandas as pd
import time
from gtts import gTTS 
from playsound import playsound



class Check_module(FunctionBase):
    yolo_model = yoloVIDVIP()
    def __init__(self):
        self.t = None
        self.cap = cv2.VideoCapture(0)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.object_position = None
    
    def execute(self, t):
        self.t = t
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            print('フレームを取得できません')


class Check_BB(Check_module):
    def __init__(self):
        super().__init__()
        self.lower_yellow = np.array([8, 75, 181])
        self.upper_yellow = np.array([58, 225, 281])

    def execute(self, module, t):
        '''一番黄色がある範囲に絞ってYOLOを適用'''
        super().execute(t)
        num_yellow = np.array([])
        for i in range(4):
            area = self.frame[int(self.height*5/12) : int(self.height*11/12), int(i*self.width/4) : int((i+1)*self.width/4)]
            hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
            num_yellow = np.append(num_yellow, np.count_nonzero(yellow_mask==255))
        max_idx = np.argmax(num_yellow)
        img = eq_to_pers(self.frame, np.pi/2, ((max_idx-2)/2 + 1/4)*np.pi, np.pi/6, 480, 480)
        img_name = 'temporary.jpg'
        cv2.imwrite(img_name, img)
        result = self.yolo_model.detect(img_name.encode(), ['braille_block'], max_num=1, thresh=.7)
        if not result:
            # print('点字ブロックはありません')
            return
        module.reset_activation()
        add_status = list(module.add_statuses)[0]
        add_status.activate(self.t)
        e2c_data = np.pi/2, (max_idx-2)*np.pi/2 + np.pi/4, np.pi/6, 480, 480
        add_status.set_object_position(e2c_data, result[0][2])


class Check_crosswalk(Check_module):
    def __init__(self):
        super().__init__()

    def execute(self, module, t):
        '''前方2方向に対してYOLO実行'''
        super().execute(t)
        max_score = -float('inf')
        e2c_data = None
        BB_data = None
        for i in range(2):
            img = eq_to_pers(self.frame, np.pi/2, (i/2-1/4)*np.pi, np.pi/6, 480, 480)
            img_name = 'temporary.jpg'
            cv2.imwrite(img_name, img)
            results = self.yolo_model.detect(img_name.encode(), ['crosswalk'], max_num=1, thresh=.7)
            for result in results:
                if result[2][2]*result[2][3] > max_score:
                    max_score = result[2][2]*result[2][3]
                    e2c_data = np.pi/2, i*np.pi/2-np.pi/4, np.pi/6, 480, 480
                    BB_data = result[2]
        # print(dir(module))
        # print(module.name)
        # print(BB_data)
        if not BB_data:
            # print('横断歩道はありません)
            return
        module.reset_activation()
        add_status = list(module.add_statuses)[0]
        add_status.activate(self.t)
        add_status.set_object_position(e2c_data, BB_data)


class Check_pedestrian_TL(Check_module):
    def __init__(self):
        super().__init__()

    def execute(self, module, t):
        '''横断歩道の位置から推測し、YOLO実行'''
        super().execute(t)
        for status in module.condition_statuses:
            if status.name == 'is crosswalk':
                is_crosswalk = status
        cw_e2c_data, cw_BB = is_crosswalk.object_position
        cw_e2c_data, cw_BB = is_crosswalk.modify_bb_data(is_crosswalk.object_position)
        cw_fov, cw_u, cw_v, cw_h, cw_w = cw_e2c_data
        cw_BB_x, cw_BB_y, cw_BB_w, cw_BB_h = cw_BB

        new_fov = cw_fov/3
        new_u = cw_u + cw_fov*((cw_BB_x-cw_w/2)/cw_w)
        new_v = cw_v - np.pi/9
        new_h = cw_h
        new_w = cw_w
        img = eq_to_pers(self.frame, new_fov, new_u, new_v, new_h, new_w)
        img_name = 'temporary.jpg'
        cv2.imwrite(img_name, img)
        result = self.yolo_model.detect(img_name.encode(), ['traffic_light', 'signal_red', 'signal_blue'], max_num=1, thresh=.7)
        if not result:
            # print('歩行者用信号機はありません')
            return
        module.reset_activation()
        add_status = list(module.add_statuses)[0]
        add_status.activate(self.t)
        e2c_data = new_fov, new_u, new_v, new_h, new_w
        add_status.set_object_position(e2c_data, result[0][2])


class Check_color_of_pedestrian_TL(Check_module):
    def __init__(self):
        super().__init__()

    def execute(self, module, t):
        """
        check_pedestrian_TLで認識した位置の周りにYOLOを適用して、sigral_red(blue)が出ればOK。
        もしtraffic lightって出たら、その範囲の色から青か赤かを判断。
        """
        super().execute(t)
        for status in module.add_statuses:
            if status.name == 'pedestrian TL is red':
                signal_red = status
            elif status.name == 'pedestrian TL is blue':
                signal_blue = status
        is_pedestrian_TL = list(module.condition_statuses)[0]
        tl_e2c_data, tl_BB = is_pedestrian_TL.modify_bb_data(is_pedestrian_TL.object_position)
        tl_fov, tl_u, tl_v, tl_h, tl_w = tl_e2c_data
        tl_BB_x, tl_BB_y, tl_BB_w, tl_BB_h = tl_BB

        ratio_h = tl_BB_w / tl_w + 0.3
        ratio_w = tl_BB_h / tl_h + 0.3
        # new_fov = tl_fov * max(ratio_h, ratio_w)
        new_fov = tl_fov
        new_u = tl_u + tl_fov*((tl_BB_x-tl_w/2)/tl_w)
        new_v = tl_v + tl_fov*((tl_BB_y-tl_h/2)/tl_h)
        new_h = tl_h
        new_w = tl_w
        img = eq_to_pers(self.frame, new_fov, new_u, new_v, new_h, new_w)
        img_name = 'temporary.jpg'
        cv2.imwrite(img_name, img)
        result = self.yolo_model.detect(img_name.encode(), ['traffic_light', 'signal_red', 'signal_blue'], max_num=1, thresh=.7)
        if not result:
            # print('歩行者用信号機はありません')
            is_pedestrian_TL.deactivate()
            return
        module.reset_activation()
        if result[0][0] == 'signal_red':
            signal_red.activate(self.t)
        elif result[0][0] == 'signal_blue':
            signal_blue.activate(self.t)
        elif result[0][0] == 'traffic_light':
            r_x, r_y, r_h, r_w = result[0][2]
            area = img[int(r_y-r_h/2):int(r_y+r_h/2), int(r_x-r_w/2):int(r_x+r_w/2)]
            hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
            hsv_mean = hsv.mean(axis=0).mean(axis=0)
            hsv_mean_red = np.array([4, 169, 203])
            hsv_mean_blue = np.array([94, 183, 132])
            dist_red = np.linalg.norm(hsv_mean - hsv_mean_red)
            dist_blue = np.linalg.norm(hsv_mean - hsv_mean_blue)
            if dist_red > dist_red:
                signal_red.activate(self.t)
            else:
                signal_blue.activate(self.t)

            
class Check_cars(Check_module):
    def __init__(self):
        super().__init__()
        self.flags = [False, False]
        self.areas = [None, None]
        # self.imgs = [None, None]
        self.thresh = 1.3

    def execute(self, module, t):
        """
        crosswalkの前にいるから左右のcarなどを検出。
        """
        super().execute(t)
        for i in range(2):
            img = eq_to_pers(eqimg=self.frame, fov=np.pi/2, u=(i*0.6-0.3)*np.pi, v=np.pi/6, out_h=480, out_w=480)
            img_name = 'temporary.jpg'
            cv2.imwrite(img_name, img)
            result = self.yolo_model.detect(img_name.encode(), ['car'], max_num=1, thresh=.5, key='area')
            if not result:
                self.flags[i] = True
                # print('車は来ていません')
            else:
                self.areas[i] = result[0][2][2] * result[0][2][3]
        # time.sleep(0.2) # 単純な処理だけで時間差は生まれる
        while not all(self.flags):
            ret, frame = self.cap.read()
            for i in range(2):
                if not self.flags[i]:
                    img = eq_to_pers(eqimg=frame, fov=np.pi/2, u=(i*0.6-0.3)*np.pi, v=np.pi/6, out_h=480, out_w=480)
                    img_name = 'temporary.jpg'
                    cv2.imwrite(img_name, img)
                    result = self.yolo_model.detect(img_name.encode(), ['car'], max_num=1, thresh=.7, key='area')
                    if not result:
                        self.flags[i] = True
                        # print('車は来ていません')
                    elif result[0][2][2]*result[0][2][3] / self.areas[i] < self.thresh:
                        self.flags[i] = True
                        # print('車は止まっています。')

        module.reset_activation()
        no_cars_approaching = list(module.add_statuses)[0]
        no_cars_approaching.activate(self.t)
            
  
class Check_steps(Check_module):
    def __init__(self):
        super().__init__()

    def execute(self, module, t):
        """
        足元をチェック
        なかったらなかったでadd listは活性化する
        """
        super().execute(t)
        img = eq_to_pers(eqimg=self.frame, fov=np.pi/2, u=0, v=np.pi/3, out_h=480, out_w=480)
        img_name = 'temporary.jpg'
        cv2.imwrite(img_name, img)
        result = self.yolo_model.detect(img_name.encode(), ['steps'], max_num=1, thresh=.7)
        module.reset_activation()
        add_status = list(module.add_statuses)[0]
        add_status.activate(self.t)
        if not result:
            # print('段差はありません')
            return
        e2c_data = np.pi/2, 0, np.pi/3, 480, 480
        add_status.set_object_position(e2c_data, result[0][2])


class Check_opposite_BB(Check_module):
    def __init__(self):
        super().__init__()
        self.lower_yellow = np.array([8, 75, 181])
        self.upper_yellow = np.array([58, 225, 281])

    def execute(self, module, t):
        super().execute(t)
        num_yellow = np.array([])
        for i in range(3):
            area = self.frame[int((i*2/9+1/3)*self.height) : int(((i+1)*2/9+1/3)*self.height), int(self.width/2-240) : int(self.width/2+240)]
            hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
            num_yellow = np.append(num_yellow, np.count_nonzero(yellow_mask==255))
        max_idx = np.argmax(num_yellow)
        img = eq_to_pers(self.frame, (max_idx/18+2/9)*np.pi, 0, (max_idx*2/9-1/18)*np.pi, 480, 480)
        img_name = 'temporary.jpg'
        cv2.imwrite(img_name, img)
        result = self.yolo_model.detect(img_name.encode(), ['braille_block'], max_num=1, thresh=.7)
        if not result:
            # print('点字ブロックはありません')
            return
        module.reset_activation()
        add_status = list(module.add_statuses)[0]
        add_status.activate(self.t)
        e2c_data = (max_idx/18+2/9)*np.pi, 0, (max_idx*2/9-1/18)*np.pi, 480, 480
        add_status.set_object_position(e2c_data, result[0][2])





class Guide_module(FunctionBase):
    def __init__(self):
        self.t = None
        
    def speech_text(self, txt):
        tts =gTTS(text=txt,lang="ja")
        tts.save("speech.mp3")
        playsound("speech.mp3")

    def execute(self, t):
        self.t = t


class Guide_to_BB(Guide_module):
    def __init__(self):
        super().__init__()

    def execute(self, module, t):
        super().execute(t)
        is_BB = list(module.condition_statuses)[0]
        dis_navi, angle_navi = is_BB.calc_object_location(is_BB.object_position)
        clock_position_navi = int(np.cos(angle_navi)*3 + 9) if angle_navi >= 0 else int(np.cos(-angle_navi)*3 + 3)
        sound_text = f"{clock_position_navi}時方向、{round(dis_navi, 1)}メートル先に、点字ブロックがあります。進んでください。"
        self.speech_text(sound_text)

        module.reset_activation()
        for status in module.add_statuses:
            status.activate(self.t)
        
        time.sleep(dis_navi) # 分速60m計算→1m1秒


class Guide_to_crosswalk(Guide_module):
    def __init__(self):
        super().__init__()

    def execute(self, module, t):
        super().execute(t)
        is_crosswalk = list(module.condition_statuses)[0]
        dis_navi, angle_navi = is_crosswalk.calc_object_location(is_crosswalk.object_position)
        clock_position_navi = int(np.cos(angle_navi)*3 + 9) if angle_navi >= 0 else int(np.cos(-angle_navi)*3 + 3)
        sound_text = f"{clock_position_navi}時方向、{round(dis_navi, 1)}メートル先に、横断歩道があります。横断歩道手前まで進んでください。"
        self.speech_text(sound_text)

        module.reset_activation()
        for status in module.add_statuses:
            status.activate(self.t)
        
        time.sleep(dis_navi)


class Enter_intersection(Guide_module):
    def __init__(self):
        super().__init__()

    def execute(self, module, t):
        super().execute(t)
        is_step_down = list(module.condition_statuses)[0]
        if is_step_down.object_position:
            dis_navi, angle_navi = is_step_down.calc_object_location(is_step_down.object_position)
            clock_position_navi = int(np.cos(angle_navi)*3 + 9) if angle_navi >= 0 else int(np.cos(-angle_navi)*3 + 3)
            sound_text = f"{clock_position_navi}時方向、{round(dis_navi, 1)}メートル先に、下り段差があります。交差点に進入してください。"
            time.sleep(dis_navi)
        else:
            sound_text = "段差はありません。交差点に進入してください。"
            time.sleep(2)
        self.speech_text(sound_text)

        module.reset_activation()
        for status in module.add_statuses:
            status.activate(self.t)


class Guide_to_opposite_BB(Guide_module):
    def __init__(self):
        super().__init__()

    def execute(self, module, t):
        super().execute(t)
        is_opposite_BB = list(module.condition_statuses)[0]
        dis_navi, angle_navi = is_opposite_BB.calc_object_location(is_opposite_BB.object_position)
        clock_position_navi = int(np.cos(angle_navi)*3 + 9) if angle_navi >= 0 else int(np.cos(-angle_navi)*3 + 3)
        sound_text = f"{clock_position_navi}時方向、{round(dis_navi, 1)}メートル先に、向かいの点字ブロックがあります。このまま点字ブロック手前まで進んでください。"
        self.speech_text(sound_text)

        module.reset_activation()
        for status in module.add_statuses:
            status.activate(self.t)
        
        time.sleep(dis_navi)


class Up_to_sidewalk(Guide_module):
    def __init__(self):
        super().__init__()

    def execute(self, module, t):
        super().execute(t)
        is_step_up = list(module.condition_statuses)[0]
        if is_step_up.object_position:
            dis_navi, angle_navi = is_step_up.calc_object_location(is_step_up.object_position)
            clock_position_navi = int(np.cos(angle_navi)*3 + 9) if angle_navi >= 0 else int(np.cos(-angle_navi)*3 + 3)
            sound_text = f"{clock_position_navi}時方向、{round(dis_navi, 1)}メートル先に、登り段差があります。歩道に上がってください。"
            time.sleep(dis_navi)
        else:
            sound_text = "段差はありません。歩道に上がってください。"
            time.sleep(2)
        self.speech_text(sound_text)

        module.reset_activation()
        for status in module.add_statuses:
            status.activate(self.t)


class Wait_signal(Guide_module):
    def __init__(self):
        super().__init__()
    
    def execute(self, module, t):
        super().execute(t)
        sound_text = "赤信号です。青に変わるまで待ちます。"
        self.speech_text(sound_text)
        
        module.reset_activation()
        for status in module.add_statuses:
            status.activate(self.t)