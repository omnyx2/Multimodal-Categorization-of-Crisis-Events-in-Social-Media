"""
@author: Chonghan Chen <chonghac@cs.cmu.edu>
"""
from os import path as osp
import os
import logging
from PIL.Image import SAVE
from torch.serialization import save
from args import get_args
from trainer import Trainer
from crisismmd_dataset import CrisisMMDataset, CrisisMMDatasetWithSSE
from mm_models import DenseNetBertMMModel, ImageOnlyModel, TextOnlyModel
import os
import numpy as np
import torch
from torch.nn.modules import activation
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import time

import nltk

# nltk는 자연어 처리를 위해 만들어진 패키지 이다. 'stopwords'는 불용어를 의미하며 개발자는 불용어를 추가 할 수 있습니다.
nltk.download('stopwords')


if __name__ == '__main__':
    # 인자를 받아서 어떤 옵션으로 프로그램을 구동할 것인지 결정한다.
    opt = get_args()
    
    # opt에서  Model_to_load 를 통해서 어떤 모델을 사용할 것인지 설정한다 초기값이 ''이므로 무조건 스트링을 넣어주어야 하는 것으로 보인다
    model_to_load = opt.model_to_load
    # opt에서  image_model_to_load 를 통해서 어떤 모델을 사용할 것인지 설정한다 초기값이 ''이므로 무조건 스트링을 넣어주어야 하는 것으로 보인다
    image_model_to_load = opt.image_model_to_load
    # opt에서 어떤 텍스트 처리 모델을 사용할 것인지 설정한다. 초기값이 ''이므로 무조건 스트링을 넣어주어야 하는 것으로 보인다.
    text_model_to_load = opt.text_model_to_load
    
    # 어떤 그래픽 처리 디바이스를 사용할 것인지 초기값은 CUDA이다.
    device = opt.device
    # 몇개의 워커를 사용하는지, 초기값은 0이다.
    num_workers = opt.num_workers
    
    # ??
    EVAL = opt.eval
    # 텐서보드를 통해서 시각화를 시도할 것인지 알아본다 초기 값은 store_ture인데 이것이 무엇을 의미하는가 <추후 보충 요구>
    USE_TENSORBOARD = opt.use_tensorboard
    # 결과값의 저장 디렉토리에 대한 결정
    SAVE_DIR = opt.save_dir
    # 모델의 이름의 초기값은 ''이다. 만약에 이름이 없다면 
    MODEL_NAME = opt.model_name if opt.model_name else str(int(time.time()))
    
    # 이미지 텍스트 모드 중에 선택
    MODE = opt.mode
    # Task를 선택, 이에따라 최종 결과물의 사이즈가 달라진다. < 추후 보충 필요 > 
    TASK = opt.task
    # 몇번 반복할 것인지 선택 < 추후 보충 필요 >
    MAX_ITER = opt.max_iter
    
    # 아웃풋 결과물
    OUTPUT_SIZE = None 
    if TASK == 'task1':
        OUTPUT_SIZE = 2
    elif TASK == 'task2':
        OUTPUT_SIZE = 8
    elif TASK == 'task2_merged':
        OUTPUT_SIZE = 6
    else:
        # Task 설정이 제대로 되지 않으면 에러가 발생한다.
        raise NotImplemented

    # The authors did not report the following values, but they tried
    # pv, pt in [10, 20000], and pv0, pt0 in [0, 1]
    WITH_SSE = opt.with_sse
    pv = opt.pv # How many times more likely do we transit to the same class
    pt = opt.pt 
    pv0 = opt.pv0  # Probability of not doing a transition
    pt0 = opt.pt0

    # General hyper parameters
    learning_rate = opt.learning_rate
    batch_size = opt.batch_size

    # Create folder for saving
    save_dir = osp.join(SAVE_DIR, MODEL_NAME)
    if not osp.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not osp.exists(save_dir):
        os.mkdir(save_dir)


    # set logger
    logging.basicConfig(filename=osp.join(save_dir, 'output_{}.log'.format(int(time.time()))), level=logging.INFO)


    train_loader, dev_loader = None, None
    if not EVAL:
        if WITH_SSE:
            train_set = CrisisMMDatasetWithSSE()
            train_set.initialize(opt, pv, pt, pv0, pt0, phase='train', cat='all',
                                 task=TASK)
        else:
            train_set = CrisisMMDataset()
            train_set.initialize(opt, phase='train', cat='all',
                                 task=TASK)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dev_set = CrisisMMDataset()
    dev_set.initialize(opt, phase='dev', cat='all',
                       task=TASK)

    dev_loader = DataLoader(
        dev_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_set = CrisisMMDataset()
    test_set.initialize(opt, phase='test', cat='all',
                        task=TASK)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    loss_fn = nn.CrossEntropyLoss()
    if MODE == 'text_only':
        model = TextOnlyModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
    elif MODE == 'image_only':
        model = ImageOnlyModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
    elif MODE == 'both':
        model = DenseNetBertMMModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
    else:
        raise NotImplemented

    # The authors did not mention configurations of SGD. We assume they did not use momentum or weight decay.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # The authors used factor=0.1, but did not mention other configs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, cooldown=0, verbose=True)

    trainer = Trainer(train_loader, dev_loader, test_loader,
                      model, loss_fn, optimizer, scheduler, eval=EVAL, device=device, tensorboard=USE_TENSORBOARD, mode=MODE)

    if model_to_load:
        model.load(model_to_load)
        logging.info("\n***********************")
        logging.info("Model Loaded!")
        logging.info("***********************\n")
    if text_model_to_load:
        model.load(text_model_to_load)
    if image_model_to_load:
        model.load(image_model_to_load)

    if not EVAL:
        logging.info("\n================Training Summary=================")
        logging.info("Training Summary: ")
        logging.info("Learning rate {}".format(learning_rate))
        logging.info("Batch size {}".format(batch_size))
        logging.info(trainer.model)
        logging.info("\n=================================================")

        trainer.train(MAX_ITER)

    else:
        logging.info("\n================Evaluating Model=================")
        logging.info(trainer.model)

        trainer.validate()
        trainer.predict()
