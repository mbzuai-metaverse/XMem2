ffmpeg -i /home/ariana/RESEARCH/XMem_baseline/output/xmem_memory/paper_results/turkish_ice_cream/our_method_frame_selector/10_frames_1_iter/overlay/frame_%06d.jpg -i \
/home/ariana/RESEARCH/XMem_baseline/output/xmem_memory/paper_results/turkish_ice_cream/our_method_human_selector/10_frames_1_iter/overlay/frame_%06d.jpg -filter_complex hstack \
our_method_10_frame_isaacc_vs_human/frame_%06d.jpg


ffmpeg -i turkish_ice_cream.MP4 -qscale:v 2 -start_number 0 -r 30 turkish_ice_cream/JPEGImages/frame_%06d.jpg
ffmpeg -i queen_car.MOV -qscale:v 2 -start_number 0 -r 30 queen_car/JPEGImages/frame_%06d.jpg

ffmpeg -pattern_type glob -i './*.jpg' -filter:v scale=720:-1:flags=lanczos,fps=24 -gifflags +transdiff dog_tail_frse10_human10_all22.gif

python interactive_demo.py --images /home/ariana/RESEARCH/Datasets/metaverse/to_process/queen_car/JPEGImages

Solucion fue 
borrar todo lo de qt, y opencv
 luego solo instalar qt y solo opencv de scratch 

#sirve 
ffmpeg -pattern_type glob -i './overlay/*.jpg' -filter:v "crop=1080:1080:(iw-1080)/2:(ih-1080)/2" ./overlay_cropped/frame_cropped_%06d.jpg

#mod start by 0 
ffmpeg -pattern_type glob -i './5_picked_frames_preload/*.jpg' -filter:v "crop=1080:1080:(iw-1080)/2:(ih-1080)/2" -start_number 0 ./overlay_cropped/frame_cropped_%06d.jpg

# al lado cropped 
#mod start by 0 
ffmpeg -pattern_type glob -i './overlay/*.jpg' -filter:v "crop=1080:1080:0:0" -start_number 0 ./overlay_cropped/frame_cropped_%06d.jpg


#mod start 0 and resize 200:200 
ffmpeg -pattern_type glob -i './overlay/*.jpg' -filter_complex "[0:v]crop=1080:1080:(iw-1080)/2:(ih-1080)/2,scale=200:200" -start_number 0 ./overlay_cropped_resized/frame_cropped_%06d.jpg
ffmpeg -pattern_type glob -i './*.jpg' -filter_complex "[0:v]crop=1080:1080:(iw-1080)/2:(ih-1080)/2" -start_number 0 ./frame_cropped_%06d.jpg

ffmpeg -framerate 30 -i /path/to/frames/%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4


ffmpeg -i /home/ariana/RESEARCH/Datasets/MOSE/selected_videos/JPEGImages/85aa3b0e/*.jpg -c:v libx264 -pix_fmt yuv420p 85aa3b0e.mp4
ffmpeg -pattern_type glob -i './5_picked_frames/*.png' -c:v libx264 -pix_fmt yuv420p  full_face_5_frames_original_xmem.mp4

#what i used before to make a gif 
ffmpeg -pattern_type glob -i './*.jpg' -filter:v scale=720:-1:flags=lanczos,fps=24 -gifflags +transdiff turkish_icecream_frse10_human_10.gif


#from MOV to MP4
ffmpeg -i vlog.mov -c:v libx264 -c:a aac -strict -2 vlog.mp4
