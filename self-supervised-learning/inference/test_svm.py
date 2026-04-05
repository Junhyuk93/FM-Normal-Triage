import joblib
import numpy as np
import sys

def check_svm_dimensions(svm_path):
    """
    SVM 모델을 로드하고 기대하는 입력 차원을 확인
    """
    print(f"Loading SVM model from: {svm_path}")
    
    try:
        # SVM 모델 로드
        svm_model = joblib.load(svm_path)
        print(f"SVM model loaded successfully")
        print(f"Model type: {type(svm_model)}")
        
        # SVM 모델 정보 출력
        if hasattr(svm_model, 'n_features_in_'):
            print(f"Expected input features (n_features_in_): {svm_model.n_features_in_}")
        
        if hasattr(svm_model, 'support_vectors_'):
            print(f"Support vectors shape: {svm_model.support_vectors_.shape}")
            print(f"Number of support vectors: {svm_model.support_vectors_.shape[0]}")
            print(f"Feature dimension: {svm_model.support_vectors_.shape[1]}")
        
        if hasattr(svm_model, 'classes_'):
            print(f"Classes: {svm_model.classes_}")
            print(f"Number of classes: {len(svm_model.classes_)}")
        
        # 다양한 차원으로 테스트해보기
        test_dimensions = [13, 39, 26, 52]  # 가능한 차원들
        
        print("\n=== Testing different input dimensions ===")
        for dim in test_dimensions:
            try:
                # 더미 데이터 생성 (1 샘플)
                dummy_data = np.random.rand(1, dim)
                
                # 예측 시도
                pred = svm_model.predict(dummy_data)
                print(f"✓ Dimension {dim}: SUCCESS - Prediction: {pred[0]}")
                
                # 확률 예측도 시도 (가능한 경우)
                if hasattr(svm_model, 'predict_proba'):
                    try:
                        proba = svm_model.predict_proba(dummy_data)
                        print(f"  Probability shape: {proba.shape}")
                        print(f"  Probabilities: {proba[0]}")
                    except:
                        print(f"  Probability prediction not available")
                        
            except Exception as e:
                print(f"✗ Dimension {dim}: FAILED - {str(e)}")
        
        # 배치 테스트 (여러 샘플)
        print("\n=== Testing batch prediction ===")
        try:
            # 올바른 차원을 찾았다면 배치 테스트
            if hasattr(svm_model, 'n_features_in_'):
                correct_dim = svm_model.n_features_in_
            elif hasattr(svm_model, 'support_vectors_'):
                correct_dim = svm_model.support_vectors_.shape[1]
            else:
                correct_dim = 39  # 기본값
            
            batch_data = np.random.rand(5, correct_dim)  # 5개 샘플
            batch_pred = svm_model.predict(batch_data)
            print(f"Batch prediction with {correct_dim} features:")
            print(f"Input shape: {batch_data.shape}")
            print(f"Output shape: {batch_pred.shape}")
            print(f"Predictions: {batch_pred}")
            
        except Exception as e:
            print(f"Batch test failed: {str(e)}")
        
        # 추가 모델 정보
        print(f"\n=== Additional model information ===")
        print(f"Model attributes: {[attr for attr in dir(svm_model) if not attr.startswith('_')]}")
        
    except Exception as e:
        print(f"Error loading SVM model: {str(e)}")
        return None
        
    return svm_model

def main():
    # SVM 모델 경로
    svm_path = "/workspace/inference/svm_weight.pickle"
    
    # 명령행 인자로 경로를 받을 수도 있음
    if len(sys.argv) > 1:
        svm_path = sys.argv[1]
    
    print("=== SVM Model Dimension Checker ===")
    print(f"Checking SVM model: {svm_path}")
    print("=" * 50)
    
    svm_model = check_svm_dimensions(svm_path)
    
    if svm_model is not None:
        print("\n=== Summary ===")
        if hasattr(svm_model, 'n_features_in_'):
            print(f"✓ SVM expects {svm_model.n_features_in_} input features")
        elif hasattr(svm_model, 'support_vectors_'):
            print(f"✓ SVM expects {svm_model.support_vectors_.shape[1]} input features")
        else:
            print("! Could not determine expected input dimension from model attributes")
            print("  Check the test results above for working dimensions")

if __name__ == "__main__":
    main()
