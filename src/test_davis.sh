python3 evaluate.py --model "vitdet-l" --frame-rates 30 --dmap-type "topk" --dirty-topk 128 --method "cstvit" --refmap-type "topk" --similar-topk 32
python3 evaluate.py --model "vitdet-l" --frame-rates 30 --dmap-type "topk" --dirty-topk 256 --method "cstvit" --refmap-type "topk" --similar-topk 64
python3 evaluate.py --model "vitdet-l" --frame-rates 30 --dmap-type "topk" --dirty-topk 512 --method "cstvit" --refmap-type "topk" --similar-topk 128
python3 evaluate.py --model "vitdet-l" --frame-rates 30 --dmap-type "topk" --dirty-topk 1024 --method "cstvit" --refmap-type "topk" --similar-topk 256
# python3 evaluate.py --model "vitdet-l" --frame-rates 30 --dmap-type "topk" --dirty-topk 256 --method "cstvit"
# python3 evaluate.py --model "vitdet-l" --frame-rates 30 --dmap-type "topk" --dirty-topk 512 --method "cstvit"
# python3 evaluate.py --model "vitdet-l" --frame-rates 30 --dmap-type "topk" --dirty-topk 1024 --method "cstvit"

# python3 evaluate.py --model "vitdet-b" --frame-rates 30 --dmap-type "threshold" --dirty-thres 30 --method "ours"

# python3 evaluate.py --model "vitdet-b" --frame-rates 100 --dmap_type "topk"
# python3 evaluate.py --model "vitdet-l" --frame-rates 100 --dmap_type "threshold"
# python3 evaluate.py --model "vitdet-l" --frame-rates 100 --dmap_type "topk"
# python3 evaluate.py --model "vitdet-h" --frame-rates 100 --dmap_type "threshold"
# python3 evaluate.py --model "vitdet-h" --frame-rates 100 --dmap_type "topk"
# python3 evaluate.py --model "vitdet-b" --frame-rates 30 --dmap_type "threshold"
# python3 evaluate.py --model "vitdet-b" --frame-rates 30 --dmap_type "topk"
# python3 evaluate.py --model "vitdet-l" --frame-rates 30 --dmap_type "threshold"
# python3 evaluate.py --model "vitdet-l" --frame-rates 30 --dmap_type "topk"
# python3 evaluate.py --model "vitdet-h" --frame-rates 30 --dmap_type "threshold"
# python3 evaluate.py --model "vitdet-h" --frame-rates 30 --dmap_type "topk"