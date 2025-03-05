import os

from ipconv.models import MaskedRCNN_ViT_B_FPN_Contexted

def main():
    
    model = MaskedRCNN_ViT_B_FPN_Contexted("cuda")
    model.load_weight("./ipconv/models/model_final_61ccd1.pkl")

    sequence_names = os.listdir("/data/DAVIS/JPEGImages/480p")

    for sequence_name in sorted(sequence_names):
        if os.path.exists(f"./output/maskedrcnn_vit_b_fpn/plain_inference/{sequence_name}"):
            continue

        # save result to file
        os.makedirs(f"./output/maskedrcnn_vit_b_fpn/plain_inference/{sequence_name}", exist_ok=True)

        try:
            log_file = open(f"./output/maskedrcnn_vit_b_fpn/plain_inference/{sequence_name}/log.txt", "w")
            log_file.write(f"Sequence: {sequence_name}\n")


            print(f"Processing sequence: {sequence_name}")
            avg_iou_gt, inference_results = model.validate_DAVIS_plain(sequence_name)
            print(f"  - Average IoU (GT): {avg_iou_gt}")
            print()

            log_file.write(f"  Plain")
            log_file.write(f"    - Average IoU (GT): {avg_iou_gt}\n")
            log_file.write("\n")
        except:
            os.rmdir(f"./output/maskedrcnn_vit_b_fpn/plain_inference/{sequence_name}")
        

if __name__ == "__main__":
    main()
