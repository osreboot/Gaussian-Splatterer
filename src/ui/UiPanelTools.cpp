#include "UiPanelTools.h"
#include "UiFrame.h"

using namespace std;

void UiPanelTools::updateSplatCount() {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    textSplatCount->SetLabel(to_string(frame->trainer->model->count) + " / " +
        to_string(frame->trainer->model->capacity) + " splats");
}

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

    buttonCamerasCapture = new wxButton(this, wxID_ANY, "Capture");
    buttonCamerasCapture->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelTools::onButtonCamerasCapture, this);
    sizerStaticTruth->Add(buttonCamerasCapture, wxSizerFlags().Expand().Border());

    textCamerasStatus = new wxStaticText(this, wxID_ANY, "[ no data ]");
    sizerStaticTruth->Add(textCamerasStatus, wxSizerFlags().Border());



    sizerStaticTrain = new wxStaticBoxSizer(wxVERTICAL, this, "3. Train Splats");
    sizer->Add(sizerStaticTrain, wxSizerFlags().Expand().Border());

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

    textSplatCount = new wxStaticText(this, wxID_ANY, to_string(frame->trainer->model->count) + " / " +
        to_string(frame->trainer->model->capacity) + " splats");
    sizerStaticTrain->Add(textSplatCount, wxSizerFlags().Border());




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

void UiPanelTools::onSpinCtrlCamerasCount(wxSpinEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->truthCameras->setCount(event.GetValue());
    spinCtrlPreviewCamera->SetRange(1, event.GetValue());
}

void UiPanelTools::onSpinCtrlCamerasDistance(wxSpinDoubleEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->truthCameras->setDistance((float)event.GetValue());
}

void UiPanelTools::onButtonCamerasCapture(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->captureTruths(*frame->truthCameras, *frame->rtx);
    textCamerasStatus->SetLabel("[ " + to_string(frame->trainer->truthFrameBuffers.size()) + " saved truth frames ]");
    buttonTrain->Enable();
    buttonTrain10->Enable();
    buttonTrain100->Enable();
    buttonTrainDensify->Enable();
}

void UiPanelTools::onButtonTrain(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->train(false);
}

void UiPanelTools::onButtonTrain10(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->train(10);
}

void UiPanelTools::onButtonTrain100(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->train(100);
}

void UiPanelTools::onButtonTrainDensify(wxCommandEvent& event) {
    UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->train(true);
    updateSplatCount();
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
