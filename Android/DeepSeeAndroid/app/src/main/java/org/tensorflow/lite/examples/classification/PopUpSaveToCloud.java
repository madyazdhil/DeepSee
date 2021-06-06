package org.tensorflow.lite.examples.classification;

import android.app.AlertDialog;
import android.app.Dialog;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.RadioGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.DialogFragment;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.engine.DiskCacheStrategy;

import org.tensorflow.lite.examples.classification.data.DataSet;

import java.util.Objects;

public class PopUpSaveToCloud extends DialogFragment {
    private final DataSet dataSet;
    private final Bitmap image;
    public static final String PRECISE = "Precise";
    public static final String ACCURATE = "Accurate";
    public static final String NOT_ACCURATE = "Not Accurate";
    private OnSaveCallback onSaveCallback;

    public void setOnSaveCallback(OnSaveCallback onSaveCallback) {
        this.onSaveCallback = onSaveCallback;
    }

    public PopUpSaveToCloud(Bitmap image, DataSet dataSet) {
        this.dataSet = dataSet;
        this.image = image;
    }

    @NonNull
    @Override
    public Dialog onCreateDialog(@Nullable Bundle savedInstanceState) {
        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        LayoutInflater inflater = requireActivity().getLayoutInflater();
        View view = inflater.inflate(R.layout.pop_up_save_to_cloud, null);
        builder.setView(view);

        ImageView imgResult = view.findViewById(R.id.img_result);
        TextView txtMlResult = view.findViewById(R.id.txt_ml_result);

        txtMlResult.setText(dataSet.getMlResult() + " (" + dataSet.getPercentResult() + ")");

        Glide.with(view)
                .load(image)
                .diskCacheStrategy(DiskCacheStrategy.RESOURCE)
                .into(imgResult);

        TextView txtView = view.findViewById(R.id.textView3);
        EditText edtResultRight = view.findViewById(R.id.edt_correct_result);

        RadioGroup rgAccuracy = view.findViewById(R.id.rg_accuracy);

        rgAccuracy.setOnCheckedChangeListener((radioGroup, id) -> {
            if (id == R.id.rb_precise) {
                dataSet.setAccuracy(PRECISE);
                txtView.setVisibility(View.GONE);
                edtResultRight.setVisibility(View.GONE);
            } else if (id == R.id.rb_accurate) {
                dataSet.setAccuracy(ACCURATE);
                txtView.setVisibility(View.GONE);
                edtResultRight.setVisibility(View.GONE);
            } else if (id == R.id.rb_not_accurate) {
                dataSet.setAccuracy(NOT_ACCURATE);
                txtView.setVisibility(View.VISIBLE);
                edtResultRight.setVisibility(View.VISIBLE);
            }
        });

        Button btnSave = view.findViewById(R.id.btn_save);
        Button btnCancel = view.findViewById(R.id.btn_cancel);

        btnSave.setOnClickListener(v -> {
            if (dataSet.getAccuracy().equals(NOT_ACCURATE)) {
                if (edtResultRight.getText().toString().isEmpty()) {
                    edtResultRight.setError("This is mandatory!");
                } else {
                    dataSet.setUserThinkResult(edtResultRight.getText().toString());
                    onSaveCallback.onClick(image, dataSet);
                    Objects.requireNonNull(getDialog()).dismiss();
                }
            } else {
                onSaveCallback.onClick(image, dataSet);
                Objects.requireNonNull(getDialog()).dismiss();
            }
        });

        btnCancel.setOnClickListener(v -> Objects.requireNonNull(getDialog()).dismiss());
        return builder.create();
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.pop_up_save_to_cloud, container);

        Objects.requireNonNull(getDialog()).getWindow().setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));

        return view;
    }

    public interface OnSaveCallback {
        void onClick(Bitmap image, DataSet dataSet);
    }

}
