import os
import re
import cv2
import time
import uuid
import glob
import copy
import torch
import queue
import pickle
import shutil
import threading
import subprocess
import numpy as np
from tqdm import tqdm
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.blending import get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk_utils.gfpgan_wrapper import GfpganEnhancer

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
comfy_musetalk_path = f'{comfy_path}/custom_nodes/ComfyUI-MuseTalk'


@torch.no_grad()
class Inference:
    model_loaded = False
    audio_processor = None
    vae = None
    unet = None
    pe = None
    gfpgan_enhancer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)

    frame_list_cache = {}
    coord_list_cache = {}
    mask_list_cache = {}
    mask_coords_cache = {}
    latent_list_cache = {}

    def __init__(self, avatar_id: str, batch_size: int = 4):
        self.avatar_id = avatar_id
        self.avatar_path = os.path.join(comfy_musetalk_path, f"results/avatars/{avatar_id}")
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_dir = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.batch_size = batch_size
        self.idx = 0
        self.image_cache = []
        self.init()
        self.load_model()

    @classmethod
    def load_model(cls):
        if not cls.model_loaded:
            print("Loading models...")
            cls.audio_processor, cls.vae, cls.unet, cls.pe = load_all_model()
            cls.pe = cls.pe.half()
            cls.vae.vae = cls.vae.vae.half()
            cls.unet.model = cls.unet.model.half()
            cls.gfpgan_enhancer = GfpganEnhancer()
            cls.model_loaded = True
            print("Models loaded successfully.")
        
    def init(self):
        if self.avatar_id not in self.latent_list_cache:
            if not os.path.exists(self.latents_out_path):
                raise Exception(f"latents({self.latents_out_path}) not exists!")
            self.latent_list_cache[self.avatar_id] = torch.load(self.latents_out_path)
        self.input_latent_list_cycle = self.latent_list_cache[self.avatar_id]

        if self.avatar_id not in self.coord_list_cache:
            with open(self.coords_path, 'rb') as f:
                self.coord_list_cache[self.avatar_id] = pickle.load(f)
        self.coord_list_cycle = self.coord_list_cache[self.avatar_id]

        if self.avatar_id not in self.frame_list_cache:
            if not os.path.exists(self.full_imgs_path):
                raise Exception(f"full_imgs({self.full_imgs_path}) not exists!")
            input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.frame_list_cache[self.avatar_id] = read_imgs(input_img_list)
        self.frame_list_cycle = self.frame_list_cache[self.avatar_id]

        if self.avatar_id not in self.mask_coords_cache:
            if not os.path.exists(self.mask_coords_path):
                raise Exception(f"mask({self.mask_coords_path}) not exists!")
            with open(self.mask_coords_path, 'rb') as f:
                self.mask_coords_cache[self.avatar_id] = pickle.load(f)
        self.mask_coords_list_cycle = self.mask_coords_cache[self.avatar_id]

        if self.avatar_id not in self.mask_list_cache:
            input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
            input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.mask_list_cache[self.avatar_id] = read_imgs(input_mask_list)
        self.mask_list_cycle = self.mask_list_cache[self.avatar_id]
        
    def process_frames(self, 
                       res_frame_queue,
                       video_len,
                       enhance):
        """
        Process frames, blend results, and store the numpy array in image_cache.
        """
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
      
            bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % len(self.frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception as e:
                print(f"Error resizing frame: {e}")
                continue
            mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
            if enhance:
                res_frame = self.gfpgan_enhancer.enhance(res_frame)
            
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            combine_frame = combine_frame.copy()
            combine_frame = cv2.cvtColor(combine_frame, cv2.COLOR_BGR2RGB)
            combine_frame = combine_frame.astype(np.float32) / 255.0
            self.image_cache.append(torch.tensor(combine_frame))

            self.idx += 1

    def run(self, 
            audio_path: str,  
            fps: int = 25,
            enhance: bool = False):
        """
        Run the inference process and return processed numpy arrays and audio path.
        """
        print("Start inference")
        
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        print(f"Processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)   
        res_frame_queue = queue.Queue()
        self.idx = 0
        self.image_cache.clear()  # Clear cache before processing
        
        # Create a sub-thread and start it
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, enhance))
        process_thread.start()
        
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        start_time = time.time()
        
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                        dtype=self.unet.model.dtype)
            audio_feature_batch = self.pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(latent_batch, 
                                        self.timesteps, 
                                        encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        
        # Close the queue and sub-thread after all tasks are completed
        process_thread.join()

        torch.cuda.empty_cache()
        
        print(f"Total process time of {video_num} frames including saving images = {(time.time() - start_time):.2f} seconds")

        if not self.image_cache:
            raise ValueError("Processed frames are empty. No frames were generated.")
        
        print(self.image_cache[0], self.image_cache[0].shape)
        
        return self.image_cache
