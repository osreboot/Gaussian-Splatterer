#include "UiPanelTools.h"
#include "UiFrame.h"

using namespace std;

UiPanelTools::UiPanelTools(wxWindow *parent) : wxPanel(parent) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());

    sizer = new wxBoxSizer(wxVERTICAL);



    sizerStaticInput = new wxStaticBoxSizer(wxVERTICAL, this, "1. Input Model Data");
    sizer->Add(sizerStaticInput, wxSizerFlags().Expand().Border());



    sizerStaticTruth = new wxStaticBoxSizer(wxVERTICAL, this, "2. Build Truth Data");
    sizer->Add(sizerStaticTruth, wxSizerFlags().Expand().Border());

    auto textCtrlCamerasCount = new wxStaticText(this, wxID_ANY, "Perspective Number");
    sizerStaticTruth->Add(textCtrlCamerasCount, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlCamerasCount = new wxSpinCtrl(this);
    spinCtrlCamerasCount->SetRange(1, 512);
    spinCtrlCamerasCount->SetValue(frame->truthCameras->getCount());
    spinCtrlCamerasCount->SetMinSize({64, -1});
    spinCtrlCamerasCount->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelTools::onSpinCtrlCamerasCount, this);
    sizerStaticTruth->Add(spinCtrlCamerasCount, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    auto textCtrlCamerasDistance = new wxStaticText(this, wxID_ANY, "Perspective Distance");
    sizerStaticTruth->Add(textCtrlCamerasDistance, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlCamerasDistance = new wxSpinCtrlDouble(this);
    spinCtrlCamerasDistance->SetRange(0.1, 20.0);
    spinCtrlCamerasDistance->SetDigits(2);
    spinCtrlCamerasDistance->SetIncrement(0.1);
    spinCtrlCamerasDistance->SetValue(frame->truthCameras->getDistance());
    spinCtrlCamerasDistance->SetMinSize({64, -1});
    spinCtrlCamerasDistance->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelTools::onSpinCtrlCamerasDistance, this);
    sizerStaticTruth->Add(spinCtrlCamerasDistance, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    auto textCtrlCamerasRotX = new wxStaticText(this, wxID_ANY, "Rotation Offset (X)");
    sizerStaticTruth->Add(textCtrlCamerasRotX, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlCamerasRotX = new wxSpinCtrlDouble(this);
    spinCtrlCamerasRotX->SetRange(0.0, 360.0);
    spinCtrlCamerasRotX->SetDigits(1);
    spinCtrlCamerasRotX->SetIncrement(1);
    spinCtrlCamerasRotX->SetValue(frame->truthCameras->getRotationOffsetX());
    spinCtrlCamerasRotX->SetMinSize({64, -1});
    spinCtrlCamerasRotX->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelTools::onSpinCtrlCamerasRotX, this);
    sizerStaticTruth->Add(spinCtrlCamerasRotX, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    auto textCtrlCamerasRotY = new wxStaticText(this, wxID_ANY, "Rotation Offset (Y)");
    sizerStaticTruth->Add(textCtrlCamerasRotY, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlCamerasRotY = new wxSpinCtrlDouble(this);
    spinCtrlCamerasRotY->SetRange(0.0, 360.0);
    spinCtrlCamerasRotY->SetDigits(1);
    spinCtrlCamerasRotY->SetIncrement(1);
    spinCtrlCamerasRotY->SetValue(frame->truthCameras->getRotationOffsetY());
    spinCtrlCamerasRotY->SetMinSize({64, -1});
    spinCtrlCamerasRotY->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelTools::onSpinCtrlCamerasRotY, this);
    sizerStaticTruth->Add(spinCtrlCamerasRotY, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    buttonCamerasRotRandom = new wxButton(this, wxID_ANY, "Randomize Offset");
    buttonCamerasRotRandom->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelTools::onButtonCamerasRotRandom, this);
    sizerStaticTruth->Add(buttonCamerasRotRandom, wxSizerFlags().Expand().Border());

    buttonCamerasCapture = new wxButton(this, wxID_ANY, "Capture");
    buttonCamerasCapture->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelTools::onButtonCamerasCapture, this);
    sizerStaticTruth->Add(buttonCamerasCapture, wxSizerFlags().Expand().Border());

    textCamerasStatus = new wxStaticText(this, wxID_ANY, "[ no data ]");
    sizerStaticTruth->Add(textCamerasStatus, wxSizerFlags().Border());



    sizerStaticTrain = new wxStaticBoxSizer(wxVERTICAL, this, "3. Train Splats");
    sizer->Add(sizerStaticTrain, wxSizerFlags().Expand().Border());

    textIterationCount = new wxStaticText(this, wxID_ANY, to_string(frame->trainer->iterations) + " iterations");
    sizerStaticTrain->Add(textIterationCount, wxSizerFlags().Border());

    textSplatCount = new wxStaticText(this, wxID_ANY, to_string(frame->trainer->model->count) + " / " +
                                                      to_string(frame->trainer->model->capacity) + " splats");
    sizerStaticTrain->Add(textSplatCount, wxSizerFlags().Border());

    buttonTrain = new wxButton(this, wxID_ANY, "Train (1x)");
    buttonTrain->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelTools::onButtonTrain, this);
    buttonTrain->Disable();
    sizerStaticTrain->Add(buttonTrain, wxSizerFlags().Expand().Border());

    buttonTrain10 = new wxButton(this, wxID_ANY, "Train (10x)");
    buttonTrain10->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelTools::onButtonTrain10, this);
    buttonTrain10->Disable();
    sizerStaticTrain->Add(buttonTrain10, wxSizerFlags().Expand().Border());

    buttonTrain100 = new wxButton(this, wxID_ANY, "Train (100x)");
    buttonTrain100->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelTools::onButtonTrain100, this);
    buttonTrain100->Disable();
    sizerStaticTrain->Add(buttonTrain100, wxSizerFlags().Expand().Border());

    buttonTrainDensify = new wxButton(this, wxID_ANY, "Train (Densify, 1x)");
    buttonTrainDensify->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelTools::onButtonTrainDensify, this);
    buttonTrainDensify->Disable();
    sizerStaticTrain->Add(buttonTrainDensify, wxSizerFlags().Expand().Border());

    auto textButtonTrainAuto = new wxStaticText(this, wxID_ANY, "Auto Train");
    sizerStaticTrain->Add(textButtonTrainAuto, wxSizerFlags().Border());

    buttonTrainAutoStart = new wxButton(this, wxID_ANY, "Start");
    buttonTrainAutoStart->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelTools::onButtonTrainAutoStart, this);
    buttonTrainAutoStart->Disable();
    sizerStaticTrain->Add(buttonTrainAutoStart, wxSizerFlags().Expand().Border());

    buttonTrainAutoStop = new wxButton(this, wxID_ANY, "Stop");
    buttonTrainAutoStop->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelTools::onButtonTrainAutoStop, this);
    buttonTrainAutoStop->Disable();
    sizerStaticTrain->Add(buttonTrainAutoStop, wxSizerFlags().Expand().Border());



    sizerStaticOutput = new wxStaticBoxSizer(wxVERTICAL, this, "4. Visualize Splats");
    sizer->Add(sizerStaticOutput, wxSizerFlags().Expand().Border());

    checkBoxPreviewCamera = new wxCheckBox(this, wxID_ANY, "View Truth Perspective");
    checkBoxPreviewCamera->Bind(wxEVT_COMMAND_CHECKBOX_CLICKED, &UiPanelTools::onCheckBoxPreviewCamera, this);
    sizerStaticOutput->Add(checkBoxPreviewCamera, wxSizerFlags().Border(wxDOWN | wxUP));

    auto textCtrlPreviewCamera = new wxStaticText(this, wxID_ANY, "View Perspective Index");
    sizerStaticOutput->Add(textCtrlPreviewCamera, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCtrlPreviewCamera = new wxSpinCtrl(this);
    spinCtrlPreviewCamera->SetRange(1, frame->truthCameras->getCount());
    spinCtrlPreviewCamera->SetValue(1);
    spinCtrlPreviewCamera->SetMinSize({64, -1});
    spinCtrlPreviewCamera->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelTools::onSpinCtrlPreviewCamera, this);
    spinCtrlPreviewCamera->Disable();
    sizerStaticOutput->Add(spinCtrlPreviewCamera, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));



    SetSizerAndFit(sizer);
}

void UiPanelTools::updateIterationCount() {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    textIterationCount->SetLabel(to_string(frame->trainer->iterations) + " iterations");
}

void UiPanelTools::updateSplatCount() {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    textSplatCount->SetLabel(to_string(frame->trainer->model->count) + " / " +
                             to_string(frame->trainer->model->capacity) + " splats");
}

void UiPanelTools::onSpinCtrlCamerasCount(wxSpinEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->truthCameras->setCount(event.GetValue());
    spinCtrlPreviewCamera->SetRange(1, event.GetValue());
}

void UiPanelTools::onSpinCtrlCamerasDistance(wxSpinDoubleEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->truthCameras->setDistance((float)event.GetValue());
}

void UiPanelTools::onSpinCtrlCamerasRotX(wxSpinDoubleEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->truthCameras->setRotationOffset((float)event.GetValue(), frame->truthCameras->getRotationOffsetY());
}

void UiPanelTools::onSpinCtrlCamerasRotY(wxSpinDoubleEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->truthCameras->setRotationOffset(frame->truthCameras->getRotationOffsetX(), (float)event.GetValue());
}

void UiPanelTools::onButtonCamerasRotRandom(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->truthCameras->setRotationOffset(((float)rand() / (float)RAND_MAX) * 360.0f, ((float)rand() / (float)RAND_MAX) * 360.0f);
    spinCtrlCamerasRotX->SetValue(frame->truthCameras->getRotationOffsetX());
    spinCtrlCamerasRotY->SetValue(frame->truthCameras->getRotationOffsetY());
}

void UiPanelTools::onButtonCamerasCapture(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->captureTruths(*frame->truthCameras, *frame->rtx);
    textCamerasStatus->SetLabel("[ " + to_string(frame->trainer->truthFrameBuffers.size()) + " saved truth frames ]");
    if(!frame->autoTraining) {
        buttonTrain->Enable();
        buttonTrain10->Enable();
        buttonTrain100->Enable();
        buttonTrainDensify->Enable();
        buttonTrainAutoStart->Enable();
    }
}

void UiPanelTools::onButtonTrain(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->train(false);
    updateIterationCount();
}

void UiPanelTools::onButtonTrain10(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->train(10);
    updateIterationCount();
}

void UiPanelTools::onButtonTrain100(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->train(100);
    updateIterationCount();
}

void UiPanelTools::onButtonTrainDensify(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->train(true);
    updateSplatCount();
}

void UiPanelTools::onButtonTrainAutoStart(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->autoTraining = true;
    buttonCamerasCapture->Disable();
    buttonTrain->Disable();
    buttonTrain10->Disable();
    buttonTrain100->Disable();
    buttonTrainDensify->Disable();
    buttonTrainAutoStart->Disable();
    buttonTrainAutoStop->Enable();
}

void UiPanelTools::onButtonTrainAutoStop(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->autoTraining = false;
    buttonCamerasCapture->Enable();
    buttonTrain->Enable();
    buttonTrain10->Enable();
    buttonTrain100->Enable();
    buttonTrainDensify->Enable();
    buttonTrainAutoStart->Enable();
    buttonTrainAutoStop->Disable();
}

void UiPanelTools::onCheckBoxPreviewCamera(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    if(event.IsChecked()) {
        frame->truthCameras->previewPerspective = spinCtrlPreviewCamera->GetValue() - 1;
        spinCtrlPreviewCamera->Enable();
    } else {
        frame->truthCameras->previewPerspective = -1;
        spinCtrlPreviewCamera->Disable();
    }
}

void UiPanelTools::onSpinCtrlPreviewCamera(wxSpinEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->truthCameras->previewPerspective = event.GetValue() - 1;
}
