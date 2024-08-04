import argparse  
import pandas as pd  
import joblib  
from src.core.model_interpretability import ModelInterpreter  

def main():  
    parser = argparse.ArgumentParser(description="Model Interpretation CLI")  
    parser.add_argument("model_path", help="Path to the saved model file")  
    parser.add_argument("data_path", help="Path to the dataset CSV file")  
    parser.add_argument("--output", "-o", help="Output directory for saved plots", default="output")  
    parser.add_argument("--feature", "-f", help="Feature to analyze", required=True)  
    parser.add_argument("--method", "-m", choices=["pdp", "feature_importance"], help="Analysis method", required=True)  
    parser.add_argument("--target_class", "-t", type=int, help="Target class for multiclass classification", default=None)  
    
    args = parser.parse_args()  

    # Load model and data  
    model = joblib.load(args.model_path)  
    data = pd.read_csv(args.data_path)  
    X = data.drop("target", axis=1)  
    y = data["target"]  

    interpreter = ModelInterpreter(model, X, y)  

    if args.method == "pdp":  
        interpreter.plot_partial_dependence([args.feature], save_path=f"{args.output}/pdp_{args.feature}.png", target_class=args.target_class)  
    elif args.method == "feature_importance":  
        interpreter.plot_feature_importance(save_path=f"{args.output}/feature_importance.png")  

    print(f"Analysis complete. Results saved in {args.output} directory.")  

if __name__ == "__main__":  
    main()