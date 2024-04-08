from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier, ExtremelyFastDecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skmultiflow.drift_detection.adwin import ADWIN
from DataInput import *

models = [
    AdaptiveRandomForestClassifier(drift_detection_method=ADWIN(delta=0.001)),
    HoeffdingAdaptiveTreeClassifier(),
    ExtremelyFastDecisionTreeClassifier()
]

for model in models:
    for i in range(100):
        X_train_batch, y_train_batch = stream.next_sample(50)
        model.partial_fit(X_train_batch, y_train_batch)

        # Verify drift detection
        if i % 100 == 0:
          if hasattr(model, 'drift_detection_method') and model.drift_detection_method.detected_change():
             # Reset to keep monitoring
            model.reset()

    # Metrics
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    
    print(f"Métricas para o modelo {type(model).__name__}:")
    print(f"Acurácia: {accuracy:.3f}")
    print(f"Precisão: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print("----------------------------------------------")