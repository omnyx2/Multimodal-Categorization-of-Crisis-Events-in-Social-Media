import argparse


def get_args():

    # 기본적인 설정을 도와주는 친구들이다
    parser = argparse.ArgumentParser()


    # -------------- Important configs --------------- #
    # mode옵션에는 이미지만, 텍스트만, 둘다가 존재한다
    parser.add_argument('--mode', choices=['both', 'image_only', 'text_only'])
    # task 옵션에는 task1, task2, 그리고 task2_merged 존재한다
    parser.add_argument('--task', choices=['task1', 'task2', 'task2_merged'])
    # learning rate는 기본적으로 작게 시작하고 값은 float형으로 시작한다
    parser.add_argument('--learning_rate', default=2e-3, type=float)
    # batch_size를 설정 가능하며 기본은 64, 정수로 설정된다.
    parser.add_argument('--batch_size', default=64, type=int)
    # 결과가 저장되어 나올 디렉토리를 설정한다. 결과는 스트링으로 나온다
    parser.add_argument('--save_dir', default='./output', type=str)
    # 사용할 모델 이름 설정, model_name 초기값은 ''이다
    parser.add_argument('--model_name', default='', type=str)
    # ??
    parser.add_argument('--consistent_only', action='store_true')
    # ?? sse는 뭔데 ??
    parser.add_argument('--with_sse', action='store_true')

    # only used when with_sse set
    parser.add_argument('--pv', default=1000, type=int)
    parser.add_argument('--pt', default=1000, type=int)
    parser.add_argument('--pv0', default=0.3, type=float)
    parser.add_argument('--pt0', default=0.7, type=float)

    # Loading model 
    parser.add_argument('--model_to_load', default='')
    parser.add_argument('--image_model_to_load', default='')
    parser.add_argument('--text_model_to_load', default='')

    # 반복 이터레이션 횟수 설정가능
    parser.add_argument('--max_iter', default=300, type=int)

    # -------------- Default ones --------------- #
    # Running configs
    
    # 아래 코드들은 메인코드를 참고하면서 조금 더 뒤져봐야할 듯하다
    # Run flag
    parser.add_argument('--debug', action='store_true')

    
    parser.add_argument('--eval', action='store_true')
    # 텐서보드를 사용할지 말지를 결정한다. 
    parser.add_argument('--use_tensorboard', action='store_true')

    # System configs
    # device는 어떤 GPU 개발 툴을 사용할 것인가를 설정하는 것이다. 여기서는 CUDA 사용 사실상 CPU는 ㅅ사용불가이므로 <이부분 검토요청>
    parser.add_argument('--device', default='cuda')
    # 정확히 워커가 어떤 것인지 모르겟다...
    parser.add_argument('--num_workers', default=0, type=int)


    # data processing
    parser.add_argument('--load_size', default=228, type=int)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--max_dataset_size', default=2147483648, type=int)


    
    return parser.parse_args()
