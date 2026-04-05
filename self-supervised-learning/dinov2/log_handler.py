import argparse
import wandb 
import json
import time
import os

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def watch_json(args, interval=2):
    wandb.login(key=args.wandb_api_key)
    wandb.init(
        project = 'FM_SSL',
        name = 'first_test',
    )


    previous_data = []
    
    if os.path.exists(args.file_path):
        print(args.file_path)
        previous_data = load_json(args.file_path)['iteration']
    
    while True:
        try:
            current_data = load_json(args.file_path)['iteration']
            
            # 파일이 변경되었는지 비교
            if len(previous_data) != len(current_data):
                print("JSON 파일이 변경되었습니다!")
                
                # 변경된 내용을 출력
                print("변경된 내용:")
                cur_log = current_data[-1]
                print(cur_log)
                print()
                wandb.log({ 'lr' : cur_log["lr"],
                           'wd' : cur_log["wd"],
                           'mom' : cur_log["mom"], 
                           'last_layer_lr' : cur_log["last_layer_lr"],
                           'current_batch_size': cur_log["current_batch_size"],
                           'total_loss': cur_log["total_loss"],
                           'dino_local_crops_loss': cur_log["dino_local_crops_loss"],
                           'dino_global_crops_loss': cur_log["dino_global_crops_loss"],
                           'koleo_loss': cur_log["koleo_loss"],
                           'ibot_loss': cur_log["ibot_loss"],
                           })

                # 이전 데이터를 현재 데이터로 갱신
                previous_data = current_data
            
        except json.JSONDecodeError:
            print("JSON 파일을 파싱하는 중 오류가 발생했습니다.")
        except FileNotFoundError:
            print(f"{args.file_path} 파일을 찾을 수 없습니다.")


        # 지정된 간격만큼 대기
        time.sleep(interval)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='/home/workspace/self-supervised-learning/dinov2/results/training_metrics.json')
    parser.add_argument("--wandb_api_key", type=str, default='258fa8b4731cc3238006d20d353bc639e463b78f') 
    parser.add_argument("--interval", type=int, default='2') 
    args = parser.parse_args()



    # 파일 변경 감시 시작
    watch_json(args, interval=args.interval)
