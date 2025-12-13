# FINAL MODEL (LINEAR SVM)


from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib
import json

print("Training FINAL MODEL: Linear SVM...")

# Train the selected best model
final_model = LinearSVC(class_weight='balanced')
final_model.fit(X_train_final, y_train)

print("\n=== FINAL MODEL EVALUATION (Linear SVM) ===\n")
y_pred_final = final_model.predict(X_test_final)
print(classification_report(y_test, y_pred_final, digits=3))

# Save final artifacts
joblib.dump(final_model, "final_fake_job_detector.pkl")
joblib.dump(tfidf, "final_tfidf_vectorizer.pkl")
joblib.dump(scaler, "final_numeric_scaler.pkl")
joblib.dump(SUSPICIOUS_KEYWORDS, "final_suspicious_keywords.pkl")
joblib.dump(num_feats, "final_numeric_feature_names.pkl")

print("\nSaved final model + vectorizer + scaler + metadata.")

# Save metadata JSON for app compatibility
meta_final = {
    "model": "LinearSVC",
    "numeric_feature_names": num_feats,
    "suspicious_keywords": SUSPICIOUS_KEYWORDS
}

with open("final_meta.json", "w") as f:
    json.dump(meta_final, f)

print("Final Phase 3 artifacts ready for deployment.")
