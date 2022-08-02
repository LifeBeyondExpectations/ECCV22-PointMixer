ngc batch run \
 --instance dgx1v.32g.1.norm \
 --commandline "cd /root/ECCV22-PointMixer/sem_seg/script; sh run_nvidia_scannet_00.sh " \
 --name "ml-model.pointmixer.exempt-scannet-00" \
 --image nvcr.io/nvidian/pmix:cuda11.1-scannet \
 --ace nv-us-west-2 \
 --result /root/PointMixerSemSeg/ 

ngc batch run \
 --instance dgx1v.32g.1.norm \
 --commandline "cd /root/ECCV22-PointMixer/sem_seg/script; sh run_nvidia_scannet_01.sh " \
 --name "ml-model.pointmixer.exempt-scannet-01" \
 --image nvcr.io/nvidian/pmix:cuda11.1-scannet \
 --ace nv-us-west-2 \
 --result /root/PointMixerSemSeg/ 

ngc batch run \
 --instance dgx1v.32g.1.norm \
 --commandline "cd /root/ECCV22-PointMixer/sem_seg/script; sh run_nvidia_scannet_02.sh " \
 --name "ml-model.pointmixer.exempt-scannet-02" \
 --image nvcr.io/nvidian/pmix:cuda11.1-scannet \
 --ace nv-us-west-2 \
 --result /root/PointMixerSemSeg/ 