#include "UiPanelToolsTruth.h"

#include "ui/UiFrame.h"
#include "ui/UiPanelViewInput.h"
#include "ui/tools/UiPanelToolsView.h"
#include "ui/tools/UiPanelToolsTrain.h"
#include "Camera.h"
#include "Project.h"
#include "Trainer.cuh"

Project& UiPanelToolsTruth::getProject() const {
    return *frame.project;
}

UiPanelToolsTruth::UiPanelToolsTruth(wxWindow* parent, UiFrame& frame) : wxPanel(parent), frame(frame) {
    wxBoxSizer* sizer = new wxBoxSizer(wxHORIZONTAL);



    sizerSphere1 = new wxStaticBoxSizer(wxHORIZONTAL, this, "Truth Camera Sphere 1");
    sizer->Add(sizerSphere1, wxSizerFlags().Border());

    wxBoxSizer* sizerSphere1_1 = new wxBoxSizer(wxVERTICAL);
    sizerSphere1->Add(sizerSphere1_1, wxSizerFlags().Border());
    wxBoxSizer* sizerSphere1_2 = new wxBoxSizer(wxVERTICAL);
    sizerSphere1->Add(sizerSphere1_2, wxSizerFlags().Border());
    wxBoxSizer* sizerSphere1_3 = new wxBoxSizer(wxVERTICAL);
    sizerSphere1->Add(sizerSphere1_3, wxSizerFlags().Border());

    sizerSphere1_1->Add(new wxStaticText(this, wxID_ANY, "Count"));
    spinSphere1Count = new wxSpinCtrl(this, S1_COUNT);
    spinSphere1Count->SetRange(1, 512);
    spinSphere1Count->SetMinSize({64, -1});
    sizerSphere1_1->Add(spinSphere1Count);

    sizerSphere1_1->Add(new wxStaticText(this, wxID_ANY, "Distance"), wxSizerFlags().Border(wxUP));
    spinSphere1Distance = new wxSpinCtrlDouble(this, S1_DISTANCE);
    spinSphere1Distance->SetRange(0.1, 40.0);
    spinSphere1Distance->SetDigits(2);
    spinSphere1Distance->SetIncrement(0.1);
    spinSphere1Distance->SetMinSize({64, -1});
    sizerSphere1_1->Add(spinSphere1Distance);

    sizerSphere1_3->Add(new wxStaticText(this, wxID_ANY, "FOV (Y Deg.)"));
    spinSphere1Fov = new wxSpinCtrlDouble(this, S1_FOV);
    spinSphere1Fov->SetRange(10.0, 120.0);
    spinSphere1Fov->SetDigits(1);
    spinSphere1Fov->SetIncrement(0.1);
    spinSphere1Fov->SetMinSize({64, -1});
    sizerSphere1_3->Add(spinSphere1Fov);

    sizerSphere1_2->Add(new wxStaticText(this, wxID_ANY, "Rotation (X,Y)"));
    spinSphere1RotX = new wxSpinCtrlDouble(this, S1_ROTX);
    spinSphere1RotX->SetRange(0.0, 360.0);
    spinSphere1RotX->SetDigits(1);
    spinSphere1RotX->SetIncrement(1);
    spinSphere1RotX->SetMinSize({64, -1});
    sizerSphere1_2->Add(spinSphere1RotX);
    spinSphere1RotY = new wxSpinCtrlDouble(this, S1_ROTY);
    spinSphere1RotY->SetRange(0.0, 360.0);
    spinSphere1RotY->SetDigits(1);
    spinSphere1RotY->SetIncrement(1);
    spinSphere1RotY->SetMinSize({64, -1});
    sizerSphere1_2->Add(spinSphere1RotY);



    sizerSphere2 = new wxStaticBoxSizer(wxHORIZONTAL, this, "Truth Camera Sphere 2");
    sizer->Add(sizerSphere2, wxSizerFlags().Border());

    wxBoxSizer* sizerSphere2_1 = new wxBoxSizer(wxVERTICAL);
    sizerSphere2->Add(sizerSphere2_1, wxSizerFlags().Border());
    wxBoxSizer* sizerSphere2_2 = new wxBoxSizer(wxVERTICAL);
    sizerSphere2->Add(sizerSphere2_2, wxSizerFlags().Border());
    wxBoxSizer* sizerSphere2_3 = new wxBoxSizer(wxVERTICAL);
    sizerSphere2->Add(sizerSphere2_3, wxSizerFlags().Border());

    sizerSphere2_1->Add(new wxStaticText(this, wxID_ANY, "Count"));
    spinSphere2Count = new wxSpinCtrl(this, S2_COUNT);
    spinSphere2Count->SetRange(0, 512);
    spinSphere2Count->SetMinSize({64, -1});
    sizerSphere2_1->Add(spinSphere2Count);

    sizerSphere2_1->Add(new wxStaticText(this, wxID_ANY, "Distance"), wxSizerFlags().Border(wxUP));
    spinSphere2Distance = new wxSpinCtrlDouble(this, S2_DISTANCE);
    spinSphere2Distance->SetRange(0.1, 40.0);
    spinSphere2Distance->SetDigits(2);
    spinSphere2Distance->SetIncrement(0.1);
    spinSphere2Distance->SetMinSize({64, -1});
    sizerSphere2_1->Add(spinSphere2Distance);

    sizerSphere2_3->Add(new wxStaticText(this, wxID_ANY, "FOV (Y Deg.)"));
    spinSphere2Fov = new wxSpinCtrlDouble(this, S2_FOV);
    spinSphere2Fov->SetRange(10.0, 120.0);
    spinSphere2Fov->SetDigits(1);
    spinSphere2Fov->SetIncrement(0.1);
    spinSphere2Fov->SetMinSize({64, -1});
    sizerSphere2_3->Add(spinSphere2Fov);

    sizerSphere2_2->Add(new wxStaticText(this, wxID_ANY, "Rotation (X,Y)"));
    spinSphere2RotX = new wxSpinCtrlDouble(this, S2_ROTX);
    spinSphere2RotX->SetRange(0.0, 360.0);
    spinSphere2RotX->SetDigits(1);
    spinSphere2RotX->SetIncrement(1);
    spinSphere2RotX->SetMinSize({64, -1});
    sizerSphere2_2->Add(spinSphere2RotX);
    spinSphere2RotY = new wxSpinCtrlDouble(this, S2_ROTY);
    spinSphere2RotY->SetRange(0.0, 360.0);
    spinSphere2RotY->SetDigits(1);
    spinSphere2RotY->SetIncrement(1);
    spinSphere2RotY->SetMinSize({64, -1});
    sizerSphere2_2->Add(spinSphere2RotY);



    sizerControls = new wxBoxSizer(wxVERTICAL);
    sizer->Add(sizerControls);

    sizerControls->Add(new wxStaticText(this, wxID_ANY, "Truth RT Samples"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinRtSamples = new wxSpinCtrl(this);
    spinRtSamples->SetRange(1, 200000);
    spinRtSamples->SetMinSize({64, -1});
    spinRtSamples->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsTruth::onSpinRtSamples, this);
    sizerControls->Add(spinRtSamples, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    buttonCapture = new wxButton(this, wxID_ANY, "Capture Truth");
    buttonCapture->SetMinSize({-1, 40});
    buttonCapture->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsTruth::onButtonCapture, this);
    sizerControls->Add(buttonCapture, wxSizerFlags().Expand().Border(wxUP | wxLEFT | wxRIGHT));

    buttonRandomRotate = new wxButton(this, wxID_ANY, "Randomize Offset");
    buttonRandomRotate->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsTruth::onButtonRandomRotate, this);
    sizerControls->Add(buttonRandomRotate, wxSizerFlags().Expand().Border(wxDOWN | wxLEFT | wxRIGHT));



    Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsTruth::onSpin, this);
    Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelToolsTruth::onSpinDouble, this);



    SetSizerAndFit(sizer);
}

void UiPanelToolsTruth::refreshProject() {
    spinSphere1Count->SetValue(getProject().sphere1.count);
    spinSphere1Distance->SetValue(getProject().sphere1.distance);
    spinSphere1Fov->SetValue(getProject().sphere1.fovDeg);
    spinSphere1RotX->SetValue(getProject().sphere1.rotX);
    spinSphere1RotY->SetValue(getProject().sphere1.rotY);
    spinSphere2Count->SetValue(getProject().sphere2.count);
    spinSphere2Distance->SetValue(getProject().sphere2.distance);
    spinSphere2Fov->SetValue(getProject().sphere2.fovDeg);
    spinSphere2RotX->SetValue(getProject().sphere2.rotX);
    spinSphere2RotY->SetValue(getProject().sphere2.rotY);

    spinRtSamples->SetValue(getProject().rtSamples);
}

void UiPanelToolsTruth::onSpin(wxSpinEvent& event) {
    switch(event.GetId()) {
        case CameraSphereIds::S1_COUNT: getProject().sphere1.count = event.GetValue(); break;
        case CameraSphereIds::S2_COUNT: getProject().sphere2.count = event.GetValue(); break;
    }
    frame.panelToolsView->refreshCameraCount();
}

void UiPanelToolsTruth::onSpinDouble(wxSpinDoubleEvent& event) {
    switch(event.GetId()) {
        case CameraSphereIds::S1_DISTANCE: getProject().sphere1.distance = (float)event.GetValue(); break;
        case CameraSphereIds::S1_FOV: getProject().sphere1.fovDeg = (float)event.GetValue(); break;
        case CameraSphereIds::S1_ROTX: getProject().sphere1.rotX = (float)event.GetValue(); break;
        case CameraSphereIds::S1_ROTY: getProject().sphere1.rotY = (float)event.GetValue(); break;
        case CameraSphereIds::S2_DISTANCE: getProject().sphere2.distance = (float)event.GetValue(); break;
        case CameraSphereIds::S2_FOV: getProject().sphere2.fovDeg = (float)event.GetValue(); break;
        case CameraSphereIds::S2_ROTX: getProject().sphere2.rotX = (float)event.GetValue(); break;
        case CameraSphereIds::S2_ROTY: getProject().sphere2.rotY = (float)event.GetValue(); break;
        default: break;
    }
}

void UiPanelToolsTruth::onSpinRtSamples(wxSpinEvent& event) {
    getProject().rtSamples = event.GetValue();
}

void UiPanelToolsTruth::onButtonCapture(wxCommandEvent& event) {
    frame.trainer->captureTruths(getProject(), *frame.rtx);
    frame.panelInput->refreshText();
    if(!frame.autoTraining) frame.panelToolsTrain->Enable();
}

void UiPanelToolsTruth::onButtonRandomRotate(wxCommandEvent& event) {
    spinSphere1RotX->SetValue(getProject().sphere1.rotX = ((float)rand() / (float)RAND_MAX) * 360.0f);
    spinSphere1RotY->SetValue(getProject().sphere1.rotY = ((float)rand() / (float)RAND_MAX) * 360.0f);
    spinSphere2RotX->SetValue(getProject().sphere2.rotX = ((float)rand() / (float)RAND_MAX) * 360.0f);
    spinSphere2RotY->SetValue(getProject().sphere2.rotY = ((float)rand() / (float)RAND_MAX) * 360.0f);
}
