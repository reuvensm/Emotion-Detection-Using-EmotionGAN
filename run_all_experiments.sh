
# resnet 18
python train_model.py --net_name resnet18 --exp_name Regular --epochs 100 --emotion_transform_prob 0 --fine_tune True
python train_model.py --net_name resnet18 --exp_name Augmented --epochs 100 --emotion_transform_prob 0.8 --fine_tune True
python train_model.py --net_name resnet18 --exp_name Expanded --epochs 100 --fine_tune True --use_expanded_dataset True

# densenet201
python train_model.py --net_name densenet201 --exp_name Regular --epochs 100 --emotion_transform_prob 0 --fine_tune True
python train_model.py --net_name densenet201 --exp_name Augmented --epochs 100 --emotion_transform_prob 0.8 --fine_tune True
python train_model.py --net_name densenet201 --exp_name Expanded --epochs 100 --fine_tune True --use_expanded_dataset True

#resnet18 BYOL
python train_byol_model.py --net_name resnet18 --exp_name BYOL --epochs 30  --fine_tune True
python train_model.py --net_name resnet18 --exp_name BYOL --epochs 100 --load_byol True --emotion_transform_prob 0 --fine_tune True

#densenet201 BYOL
python train_byol_model.py --net_name densenet201 --exp_name BYOL --epochs 30  --fine_tune True
python train_model.py --net_name densenet201 --exp_name BYOL --epochs 100 --load_byol True --emotion_transform_prob 0 --fine_tune True
