import argparse, json, os
from .config import CFG
from .train import train, prepare_datasets
from .evaluate import test_and_report
from .infer import infer_on_video, infer_on_folder, load_model


def main():
    parser = argparse.ArgumentParser(description="Hybrid Deepfake Detector")
    parser.add_argument('--mode', choices=['train','test','infer-one','infer-folder'], required=True)
    parser.add_argument('--model', type=str)
    parser.add_argument('--video', type=str)
    parser.add_argument('--folder', type=str)
    args = parser.parse_args()

    if args.mode=='train':
        train(CFG)

    elif args.mode=='test':
        assert args.model and os.path.isfile(args.model), "Provide --model for test"
        model = load_model(args.model)
        _, _, ds_te = prepare_datasets(CFG)
        report, cm = test_and_report(model, ds_te, CFG)
        print(json.dumps(report, indent=2))

    elif args.mode=='infer-one':
        assert args.model and os.path.isfile(args.model), "Provide --model for infer-one"
        assert args.video and os.path.isfile(args.video), "Provide --video for infer-one"
        model = load_model(args.model)
        print(json.dumps({"video": args.video, "prob_fake": infer_on_video(model, args.video, CFG)}, indent=2))

    elif args.mode=='infer-folder':
        assert args.model and os.path.isfile(args.model), "Provide --model for infer-folder"
        assert args.folder and os.path.isdir(args.folder), "Provide --folder for infer-folder"
        model = load_model(args.model)
        print(json.dumps(infer_on_folder(model, args.folder, CFG), indent=2))

if __name__ == '__main__':
    main()
