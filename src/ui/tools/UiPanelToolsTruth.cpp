#include "UiPanelToolsTruth.h"

UiPanelToolsTruth::UiPanelToolsTruth(wxWindow* parent) : wxPanel(parent) {
    sizer = new wxStaticBoxSizer(wxHORIZONTAL, this, "2. Build Truth Data");



    sizerSphere1 = new wxStaticBoxSizer(wxHORIZONTAL, this, "Camera Sphere 1");
    sizer->Add(sizerSphere1, wxSizerFlags().Border());

    wxBoxSizer* sizerSphere1_1 = new wxBoxSizer(wxVERTICAL);
    sizerSphere1->Add(sizerSphere1_1, wxSizerFlags().Border());
    wxBoxSizer* sizerSphere1_2 = new wxBoxSizer(wxVERTICAL);
    sizerSphere1->Add(sizerSphere1_2, wxSizerFlags().Border());

    sizerSphere1_1->Add(new wxStaticText(this, wxID_ANY, "Count"));
    spinSphere1Count = new wxSpinCtrl(this);
    spinSphere1Count->SetRange(1, 512);
    //spinSphere1Count->SetValue(frame->truthCameras->getCount());
    spinSphere1Count->SetMinSize({64, -1});
    //spinSphere1Count->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsTruth::onSpinCtrlCamerasCount, this);
    sizerSphere1_1->Add(spinSphere1Count);

    sizerSphere1_1->Add(new wxStaticText(this, wxID_ANY, "Distance"), wxSizerFlags().Border(wxUP));
    spinSphere1Distance = new wxSpinCtrlDouble(this);
    spinSphere1Distance->SetRange(0.1, 40.0);
    spinSphere1Distance->SetDigits(2);
    spinSphere1Distance->SetIncrement(0.1);
    //spinSphere1Distance->SetValue(frame->truthCameras->getDistance());
    spinSphere1Distance->SetMinSize({64, -1});
    //spinSphere1Distance->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelToolsTruth::onSpinCtrlCamerasDistance, this);
    sizerSphere1_1->Add(spinSphere1Distance);

    sizerSphere1_2->Add(new wxStaticText(this, wxID_ANY, "FOV"));
    spinSphere1Fov = new wxSpinCtrlDouble(this);
    spinSphere1Fov->SetRange(10.0, 120.0);
    spinSphere1Fov->SetDigits(1);
    spinSphere1Fov->SetIncrement(0.1);
    //spinSphere1Fov->SetValue(frame->truthCameras->getDistance());
    spinSphere1Fov->SetMinSize({64, -1});
    //spinSphere1Fov->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelToolsTruth::onSpinCtrlCamerasDistance, this);
    sizerSphere1_2->Add(spinSphere1Fov);

    sizerSphere1_2->Add(new wxStaticText(this, wxID_ANY, "Rotation (X,Y)"), wxSizerFlags().Border(wxUP));
    spinSphere1RotX = new wxSpinCtrlDouble(this);
    spinSphere1RotX->SetRange(0.0, 360.0);
    spinSphere1RotX->SetDigits(1);
    spinSphere1RotX->SetIncrement(1);
    //spinSphere1RotX->SetValue(frame->truthCameras->getRotationOffsetX());
    spinSphere1RotX->SetMinSize({64, -1});
    //spinSphere1RotX->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelToolsTruth::onSpinCtrlCamerasRotX, this);
    sizerSphere1_2->Add(spinSphere1RotX);
    spinSphere1RotY = new wxSpinCtrlDouble(this);
    spinSphere1RotY->SetRange(0.0, 360.0);
    spinSphere1RotY->SetDigits(1);
    spinSphere1RotY->SetIncrement(1);
    //spinSphere1RotY->SetValue(frame->truthCameras->getRotationOffsetY());
    spinSphere1RotY->SetMinSize({64, -1});
    //spinSphere1RotY->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelToolsTruth::onSpinCtrlCamerasRotY, this);
    sizerSphere1_2->Add(spinSphere1RotY);



    sizerSphere2 = new wxStaticBoxSizer(wxHORIZONTAL, this, "Camera Sphere 2");
    sizer->Add(sizerSphere2, wxSizerFlags().Border());

    wxBoxSizer* sizerSphere2_1 = new wxBoxSizer(wxVERTICAL);
    sizerSphere2->Add(sizerSphere2_1, wxSizerFlags().Border());
    wxBoxSizer* sizerSphere2_2 = new wxBoxSizer(wxVERTICAL);
    sizerSphere2->Add(sizerSphere2_2, wxSizerFlags().Border());

    sizerSphere2_1->Add(new wxStaticText(this, wxID_ANY, "Count"));
    spinSphere2Count = new wxSpinCtrl(this);
    spinSphere2Count->SetRange(1, 512);
    //spinSphere2Count->SetValue(frame->truthCameras->getCount());
    spinSphere2Count->SetMinSize({64, -1});
    //spinSphere2Count->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsTruth::onSpinCtrlCamerasCount, this);
    sizerSphere2_1->Add(spinSphere2Count);

    sizerSphere2_1->Add(new wxStaticText(this, wxID_ANY, "Distance"), wxSizerFlags().Border(wxUP));
    spinSphere2Distance = new wxSpinCtrlDouble(this);
    spinSphere2Distance->SetRange(0.1, 40.0);
    spinSphere2Distance->SetDigits(2);
    spinSphere2Distance->SetIncrement(0.1);
    //spinSphere2Distance->SetValue(frame->truthCameras->getDistance());
    spinSphere2Distance->SetMinSize({64, -1});
    //spinSphere2Distance->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelToolsTruth::onSpinCtrlCamerasDistance, this);
    sizerSphere2_1->Add(spinSphere2Distance);

    sizerSphere2_2->Add(new wxStaticText(this, wxID_ANY, "FOV"));
    spinSphere2Fov = new wxSpinCtrlDouble(this);
    spinSphere2Fov->SetRange(10.0, 120.0);
    spinSphere2Fov->SetDigits(1);
    spinSphere2Fov->SetIncrement(0.1);
    //spinSphere2Fov->SetValue(frame->truthCameras->getDistance());
    spinSphere2Fov->SetMinSize({64, -1});
    //spinSphere2Fov->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelToolsTruth::onSpinCtrlCamerasDistance, this);
    sizerSphere2_2->Add(spinSphere2Fov);

    sizerSphere2_2->Add(new wxStaticText(this, wxID_ANY, "Rotation (X,Y)"), wxSizerFlags().Border(wxUP));
    spinSphere2RotX = new wxSpinCtrlDouble(this);
    spinSphere2RotX->SetRange(0.0, 360.0);
    spinSphere2RotX->SetDigits(1);
    spinSphere2RotX->SetIncrement(1);
    //spinSphere2RotX->SetValue(frame->truthCameras->getRotationOffsetX());
    spinSphere2RotX->SetMinSize({64, -1});
    //spinSphere2RotX->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelToolsTruth::onSpinCtrlCamerasRotX, this);
    sizerSphere2_2->Add(spinSphere2RotX);
    spinSphere2RotY = new wxSpinCtrlDouble(this);
    spinSphere2RotY->SetRange(0.0, 360.0);
    spinSphere2RotY->SetDigits(1);
    spinSphere2RotY->SetIncrement(1);
    //spinSphere2RotY->SetValue(frame->truthCameras->getRotationOffsetY());
    spinSphere2RotY->SetMinSize({64, -1});
    //spinSphere2RotY->Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelToolsTruth::onSpinCtrlCamerasRotY, this);
    sizerSphere2_2->Add(spinSphere2RotY);



    sizerControls = new wxBoxSizer(wxVERTICAL);
    sizer->Add(sizerControls);

    buttonRandomRotate = new wxButton(this, wxID_ANY, "Randomize Offset");
    buttonRandomRotate->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsTruth::onButtonRandomRotate, this);
    sizerControls->Add(buttonRandomRotate, wxSizerFlags().Expand().Border());

    buttonCapture = new wxButton(this, wxID_ANY, "Capture");
    buttonCapture->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsTruth::onButtonCapture, this);
    sizerControls->Add(buttonCapture, wxSizerFlags().Expand().Border());

    textStatus = new wxStaticText(this, wxID_ANY, "[ no data ]");
    sizerControls->Add(textStatus, wxSizerFlags().Border());



    SetSizerAndFit(sizer);
}

void UiPanelToolsTruth::onSpinSphere1Count(wxSpinEvent& event) {
    //UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    //frame->truthCameras->setCount(event.GetValue());
    //spinCtrlPreviewCamera->SetRange(1, event.GetValue());
}

void UiPanelToolsTruth::onSpinSphere1Distance(wxSpinDoubleEvent& event) {
    //UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    //frame->truthCameras->setDistance((float)event.GetValue());
}

void UiPanelToolsTruth::onSpinSphere1RotX(wxSpinDoubleEvent& event) {
    //UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    //frame->truthCameras->setRotationOffset((float)event.GetValue(), frame->truthCameras->getRotationOffsetY());
}

void UiPanelToolsTruth::onSpinSphere1RotY(wxSpinDoubleEvent& event) {
    //UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    //frame->truthCameras->setRotationOffset(frame->truthCameras->getRotationOffsetX(), (float)event.GetValue());
}

void UiPanelToolsTruth::onButtonRandomRotate(wxCommandEvent& event) {
    /*UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->truthCameras->setRotationOffset(((float)rand() / (float)RAND_MAX) * 360.0f, ((float)rand() / (float)RAND_MAX) * 360.0f);
    spinCtrlCamerasRotX->SetValue(frame->truthCameras->getRotationOffsetX());
    spinCtrlCamerasRotY->SetValue(frame->truthCameras->getRotationOffsetY());*/
}

void UiPanelToolsTruth::onButtonCapture(wxCommandEvent& event) {
    /*UiFrame* frame = dynamic_cast<UiFrame*>(GetParent()->GetParent());
    frame->trainer->captureTruths(*frame->truthCameras, *frame->rtx);
    textCamerasStatus->SetLabel("[ " + to_string(frame->trainer->truthFrameBuffersW.size()) + " (x2) saved truth frames ]");
    if(!frame->autoTraining) {
        buttonTrain->Enable();
        buttonTrain10->Enable();
        buttonTrain100->Enable();
        buttonTrainDensify->Enable();
        buttonTrainAutoStart->Enable();
    }*/
}
