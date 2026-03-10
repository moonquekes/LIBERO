安装:
 在wsl或者window或者有图形界面的linux：
解压 到  /path/to/
conda create -n vla python=3.10.16 -y
cd /path/to/libero/setup
bash ./setup_suction_collection.sh vla /path/to/libero

采集数据
cd /path/to/libero/scripts
bash ./collect_only.sh

会出现两个窗口，wasd移动。r升f降低。空格控制吸盘开关
尽量不做多余动作，
如果需要重开按q，如果任务完成（铁块掉进框）自动保存到../data/suction_dataset/raw_hdf5，并开始下一轮采集

如果需要停止采集，在窗口开启的状态下，去命令行按ctrl c，如果直接叉掉窗口会再次出现窗口。