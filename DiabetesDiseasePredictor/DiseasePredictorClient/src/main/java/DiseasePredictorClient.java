import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Stage;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ConnectException;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class DiseasePredictorClient extends Application {

    @Override
    public void start(Stage primaryStage) {
        // Create main container with styling
        VBox mainContainer = new VBox(20);
        mainContainer.setPadding(new Insets(20));
        mainContainer.setStyle("-fx-background-color: #f5f5f5;");

        // Title
        Label titleLabel = new Label("Diabetes Predictor");
        titleLabel.setFont(Font.font("Arial", FontWeight.BOLD, 24));
        titleLabel.setTextFill(Color.DARKBLUE);

        // Form container
        GridPane formGrid = new GridPane();
        formGrid.setHgap(10);
        formGrid.setVgap(10);
        formGrid.setPadding(new Insets(20));
        formGrid.setStyle("-fx-background-color: white; -fx-border-radius: 5; -fx-border-color: #ddd;");

        // Create form fields with validation
        TextField pregnanciesField = createNumericField();
        TextField glucoseField = createNumericField();
        TextField bloodPressureField = createNumericField();
        TextField skinThicknessField = createNumericField();
        TextField insulinField = createNumericField();
        TextField bmiField = createNumericField();
        TextField diabetesPedigreeField = createNumericField();
        TextField ageField = createNumericField();

        // Add fields to grid
        addFormRow(formGrid, "Pregnancies:", pregnanciesField, 0);
        addFormRow(formGrid, "Glucose (mg/dL):", glucoseField, 1);
        addFormRow(formGrid, "Blood Pressure (mmHg):", bloodPressureField, 2);
        addFormRow(formGrid, "Skin Thickness (mm):", skinThicknessField, 3);
        addFormRow(formGrid, "Insulin (μU/ml):", insulinField, 4);
        addFormRow(formGrid, "BMI:", bmiField, 5);
        addFormRow(formGrid, "Diabetes Pedigree Function:", diabetesPedigreeField, 6);
        addFormRow(formGrid, "Age:", ageField, 7);

        // Button to submit the data
        Button predictButton = new Button("Predict Diabetes");
        predictButton.setStyle("-fx-background-color: #4a90e2; -fx-text-fill: white; -fx-font-weight: bold;");
        predictButton.setPrefHeight(40);
        predictButton.setMaxWidth(Double.MAX_VALUE);

        // Result display
        Label resultTitle = new Label("Prediction Result:");
        resultTitle.setFont(Font.font("Arial", FontWeight.BOLD, 16));

        Label resultLabel = new Label("");
        resultLabel.setFont(Font.font("Arial", FontWeight.BOLD, 18));
        resultLabel.setWrapText(true);

        VBox resultBox = new VBox(10, resultTitle, resultLabel);
        resultBox.setPadding(new Insets(20));
        resultBox.setStyle("-fx-background-color: white; -fx-border-radius: 5; -fx-border-color: #ddd;");

        // Status label for errors
        Label statusLabel = new Label("");
        statusLabel.setTextFill(Color.RED);

        // Handle button click
        predictButton.setOnAction(e -> {
            statusLabel.setText("");
            resultLabel.setText("");

            try {
                // Validate all fields
                if (!validateFields(pregnanciesField, glucoseField, bloodPressureField,
                        skinThicknessField, insulinField, bmiField, diabetesPedigreeField, ageField)) {
                    statusLabel.setText("Please fill all fields with valid numbers");
                    return;
                }

                // Collect form data
                JSONObject requestData = new JSONObject();
                requestData.put("Pregnancies", pregnanciesField.getText());  // Must match Flask
                requestData.put("Glucose", glucoseField.getText());
                requestData.put("BloodPressure", bloodPressureField.getText());
                requestData.put("SkinThickness", skinThicknessField.getText());
                requestData.put("Insulin", insulinField.getText());
                requestData.put("BMI", bmiField.getText());
                requestData.put("DiabetesPedigreeFunction", diabetesPedigreeField.getText());
                requestData.put("Age", ageField.getText());

                // Send POST request to Flask backend
                URL url = new URL("http://127.0.0.1:5000/predict");
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setDoOutput(true);
                connection.setRequestProperty("Content-Type", "application/json");

                // Set timeout (5 seconds)
                connection.setConnectTimeout(5000);
                connection.setReadTimeout(5000);

                // Send the JSON data
                try (OutputStream os = connection.getOutputStream()) {
                    byte[] input = requestData.toString().getBytes(StandardCharsets.UTF_8);
                    os.write(input, 0, input.length);
                }

                // Check response code
                int responseCode = connection.getResponseCode();
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    try (BufferedReader br = new BufferedReader(
                            new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8))) {
                        StringBuilder response = new StringBuilder();
                        String responseLine;
                        while ((responseLine = br.readLine()) != null) {
                            response.append(responseLine.trim());
                        }
                        JSONObject responseJson = new JSONObject(response.toString());
                        int prediction = responseJson.getInt("prediction");

                        // Update the result label with styling
                        if (prediction == 1) {
                            resultLabel.setText("POSITIVE for diabetes risk");
                            resultLabel.setTextFill(Color.RED);
                        } else {
                            resultLabel.setText("NEGATIVE for diabetes risk");
                            resultLabel.setTextFill(Color.GREEN);
                        }
                    }
                } else {
                    try (BufferedReader br = new BufferedReader(
                            new InputStreamReader(connection.getErrorStream(), StandardCharsets.UTF_8))) {
                        StringBuilder errorResponse = new StringBuilder();
                        String errorLine;
                        while ((errorLine = br.readLine()) != null) {
                            errorResponse.append(errorLine.trim());
                        }
                        statusLabel.setText("Server error: " + errorResponse.toString());
                    }
                }
            } catch (ConnectException ex) {
                statusLabel.setText("Could not connect to server. Is it running?");
            } catch (SocketTimeoutException ex) {
                statusLabel.setText("Request timed out. Please try again.");
            } catch (Exception ex) {
                statusLabel.setText("Error: " + ex.getMessage());
            }
        });

        // Add components to main container
        mainContainer.getChildren().addAll(
                titleLabel,
                formGrid,
                predictButton,
                resultBox,
                statusLabel
        );

        // Set up the scene and stage
        Scene scene = new Scene(mainContainer, 500, 700);
        primaryStage.setTitle("Diabetes Predictor");
        primaryStage.setScene(scene);
        primaryStage.setMinWidth(500);
        primaryStage.setMinHeight(600);
        primaryStage.show();
    }

    private TextField createNumericField() {
        TextField field = new TextField();
        field.setPrefHeight(35);
        // Add input validation to only allow numbers
        field.textProperty().addListener((observable, oldValue, newValue) -> {
            if (!newValue.matches("\\d*\\.?\\d*")) {
                field.setText(oldValue);
            }
        });
        return field;
    }

    private void addFormRow(GridPane grid, String labelText, TextField field, int row) {
        Label label = new Label(labelText);
        label.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        grid.add(label, 0, row);
        grid.add(field, 1, row);
    }

    private boolean validateFields(TextField... fields) {
        for (TextField field : fields) {
            if (field.getText().isEmpty()) {
                field.setStyle("-fx-border-color: red;");
                return false;
            } else {
                field.setStyle("");
            }
        }
        return true;
    }

    public static void main(String[] args) {
        launch(args);
    }
}