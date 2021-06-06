package org.tensorflow.lite.examples.classification.data;

public class DataSet {
    private String mlResult;
    private String userThinkResult;
    private String accuracy;
    private String percentResult;
    private String imageUrl;

    public String getMlResult() {
        return mlResult;
    }

    public void setMlResult(String mlResult) {
        this.mlResult = mlResult;
    }

    public String getUserThinkResult() {
        return userThinkResult;
    }

    public void setUserThinkResult(String userThinkResult) {
        this.userThinkResult = userThinkResult;
    }

    public String getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(String accuracy) {
        this.accuracy = accuracy;
    }

    public String getPercentResult() {
        return percentResult;
    }

    public void setPercentResult(String percentResult) {
        this.percentResult = percentResult;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public void setImageUrl(String imageUrl) {
        this.imageUrl = imageUrl;
    }

    @Override
    public String toString() {
        return "DataSet{" +
                "mlResult='" + mlResult + '\'' +
                ", userThinkResult='" + userThinkResult + '\'' +
                ", accuracy='" + accuracy + '\'' +
                ", percentResult='" + percentResult + '\'' +
                ", imageUrl='" + imageUrl + '\'' +
                '}';
    }
}
