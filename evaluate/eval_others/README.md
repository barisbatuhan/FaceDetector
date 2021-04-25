# iCartoonFace detection测评流程
1. 环境依赖
    - 机器环境: CPU机器
    - python版本: python3.6.5 
    - python依赖包: pip install numpy==1.18.1
2. 输入：
    - ./det_tmp.csv： 检测结果csv文件
3. 输出：
    - 分数
4. 运行
    - python run.py ./det_tmp.csv
3. csv格式说明，每一行的格式如下：img_name.jpg,xmin,ymin,xmax,ymax,face,score
    - personai_icartoonface_detval_00000.jpg,180,84,219,125,face,0.90
    - personai_icartoonface_detval_00000.jpg,93,87,140,125,face,0.90
    - personai_icartoonface_detval_00001.jpg,438,183,647,439,face,0.90