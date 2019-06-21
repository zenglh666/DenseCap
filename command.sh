python3 main.py --model ppn --parameters label_file=/home/zenglh/activitynet/activitynet_caption.json,visual_file=/home/zenglh/activitynet/activitynet_c3d.hdf5,language_file=/home/zenglh/activitynet/activitynet_bert.hdf5,output=/home/zenglh/DenseCap/results,train_steps=120000,decay_steps=40000,eval_steps=2000,hidden_size=256,filter_size=256,anchor_layers=7,num_mab=5,gpu=2

python3 main.py --model ppn --parameters label_file=/home/zenglh/charades/charades_caption.json,visual_file=/home/zenglh/charades/charades_visual_pca.hdf5,language_file=/home/zenglh/charades/charades_bert.hdf5,output=/home/zenglh/DenseCap/results,train_steps=60000,decay_steps=20000,eval_steps=1000,hidden_size=64,filter_size=64,anchor_layers=4,num_mab=5,gpu=0