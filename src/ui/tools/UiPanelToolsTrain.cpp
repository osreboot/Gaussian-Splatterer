#include "UiPanelToolsTrain.h"

#include "ui/UiPanelTools.h"

UiFrame& UiPanelToolsTrain::getFrame() const {
    return *dynamic_cast<UiFrame*>(GetParent()->GetParent()->GetParent());
}

Project& UiPanelToolsTrain::getProject() const {
    return *getFrame().project;
}

UiPanelToolsTrain::UiPanelToolsTrain(wxWindow* parent) : wxPanel(parent) {
    sizer = new wxStaticBoxSizer(wxHORIZONTAL, this, "3. Train Splats");



    wxBoxSizer* sizerAuto = new wxBoxSizer(wxVERTICAL);
    sizer->Add(sizerAuto, wxSizerFlags().Border());

    sizerAuto->Add(new wxStaticText(this, wxID_ANY, "Auto Train"));
    buttonAutoStart = new wxButton(this, wxID_ANY, "Start");
    buttonAutoStart->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsTrain::onButtonAutoStart, this);
    sizerAuto->Add(buttonAutoStart);
    buttonAutoStop = new wxButton(this, wxID_ANY, "Stop");
    buttonAutoStop->Disable();
    buttonAutoStop->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsTrain::onButtonAutoStop, this);
    sizerAuto->Add(buttonAutoStop);



    wxBoxSizer* sizerManual; // TODO



    wxStaticBoxSizer* sizerLr = new wxStaticBoxSizer(wxHORIZONTAL, this, "Learning Rates");
    sizer->Add(sizerLr, wxSizerFlags().Border());

    wxBoxSizer* sizerLr1 = new wxBoxSizer(wxVERTICAL);
    sizerLr->Add(sizerLr1, wxSizerFlags().Border());
    wxBoxSizer* sizerLr2 = new wxBoxSizer(wxVERTICAL);
    sizerLr->Add(sizerLr2, wxSizerFlags().Border());
    wxBoxSizer* sizerLr3 = new wxBoxSizer(wxVERTICAL);
    sizerLr->Add(sizerLr3, wxSizerFlags().Border());

    sizerLr1->Add(new wxStaticText(this, wxID_ANY, "Location"));
    spinLrLocation = new wxSpinCtrlDouble(this, LR_LOCATION);
    spinLrLocation->SetRange(0.0f, 10.0f);
    spinLrLocation->SetDigits(8);
    spinLrLocation->SetIncrement(0.0000001);
    spinLrLocation->SetMinSize({96, -1});
    sizerLr1->Add(spinLrLocation);

    sizerLr1->Add(new wxStaticText(this, wxID_ANY, "Color (SHs)"), wxSizerFlags().Border(wxUP));
    spinLrSh = new wxSpinCtrlDouble(this, LR_SH);
    spinLrSh->SetRange(0.0f, 10.0f);
    spinLrSh->SetDigits(8);
    spinLrSh->SetIncrement(0.0000001);
    spinLrSh->SetMinSize({96, -1});
    sizerLr1->Add(spinLrSh);

    sizerLr2->Add(new wxStaticText(this, wxID_ANY, "Scale"));
    spinLrScale = new wxSpinCtrlDouble(this, LR_SCALE);
    spinLrScale->SetRange(0.0f, 10.0f);
    spinLrScale->SetDigits(8);
    spinLrScale->SetIncrement(0.0000001);
    spinLrScale->SetMinSize({96, -1});
    sizerLr2->Add(spinLrScale);

    sizerLr2->Add(new wxStaticText(this, wxID_ANY, "Opacity"), wxSizerFlags().Border(wxUP));
    spinLrOpacity = new wxSpinCtrlDouble(this, LR_OPACITY);
    spinLrOpacity->SetRange(0.0f, 10.0f);
    spinLrOpacity->SetDigits(8);
    spinLrOpacity->SetIncrement(0.0000001);
    spinLrOpacity->SetMinSize({96, -1});
    sizerLr2->Add(spinLrOpacity);

    sizerLr3->Add(new wxStaticText(this, wxID_ANY, "Rotation"));
    spinLrRotation = new wxSpinCtrlDouble(this, LR_ROTATION);
    spinLrRotation->SetRange(0.0f, 10.0f);
    spinLrRotation->SetDigits(8);
    spinLrRotation->SetIncrement(0.0000001);
    spinLrRotation->SetMinSize({96, -1});
    sizerLr3->Add(spinLrRotation);



    wxBoxSizer* sizerIntervals = new wxBoxSizer(wxVERTICAL);
    sizerIntervals->SetMinSize(120, -1);
    sizer->Add(sizerIntervals, wxSizerFlags().Border());

    sizerIntervals->Add(new wxStaticText(this, wxID_ANY, "Capture Interval"));
    textIntervalCapture = new wxStaticText(this, wxID_ANY, "");
    sizerIntervals->Add(textIntervalCapture);
    spinIntervalCapture = new wxSpinCtrl(this);
    spinIntervalCapture->SetRange(0, 1000000);
    spinIntervalCapture->SetMinSize({64, -1});
    spinIntervalCapture->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsTrain::onSpinIntervalCapture, this);
    sizerIntervals->Add(spinIntervalCapture);

    sizerIntervals->Add(new wxStaticText(this, wxID_ANY, "Densify Interval"), wxSizerFlags().Border(wxUP));
    textIntervalDensify = new wxStaticText(this, wxID_ANY, "");
    sizerIntervals->Add(textIntervalDensify);
    spinIntervalDensify = new wxSpinCtrl(this);
    spinIntervalDensify->SetRange(0, 1000000);
    spinIntervalDensify->SetMinSize({64, -1});
    spinIntervalDensify->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsTrain::onSpinIntervalDensify, this);
    sizerIntervals->Add(spinIntervalDensify);



    Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelToolsTrain::onSpinParameter, this);



    SetSizerAndFit(sizer);
}

void UiPanelToolsTrain::refreshProject() {
    spinLrLocation->SetValue(getProject().lrLocation);
    spinLrSh->SetValue(getProject().lrSh);
    spinLrScale->SetValue(getProject().lrScale);
    spinLrOpacity->SetValue(getProject().lrOpacity);
    spinLrRotation->SetValue(getProject().lrRotation);

    spinIntervalCapture->SetValue(getProject().intervalCapture);
    spinIntervalDensify->SetValue(getProject().intervalDensify);
    refreshText();
}

void UiPanelToolsTrain::refreshText() {
    if (getProject().intervalCapture > 0) {
        textIntervalCapture->SetLabel(std::format(" > in {} iteration(s)",
                                                  getProject().intervalCapture - (getProject().iterations % getProject().intervalCapture)));
    } else textIntervalCapture->SetLabel(" > never");
    if (getProject().intervalDensify > 0) {
        textIntervalDensify->SetLabel(std::format(" > in {} iteration(s)",
                                                  getProject().intervalDensify - (getProject().iterations % getProject().intervalDensify)));
    } else textIntervalDensify->SetLabel(" > never");
}

void UiPanelToolsTrain::onButtonAutoStart(wxCommandEvent& event) {
    getFrame().autoTraining = true;
    buttonAutoStart->Disable();
    buttonAutoStop->Enable();
}

void UiPanelToolsTrain::onButtonAutoStop(wxCommandEvent& event) {
    getFrame().autoTraining = false;
    buttonAutoStart->Enable();
    buttonAutoStop->Disable();
}

void UiPanelToolsTrain::onSpinParameter(wxSpinDoubleEvent& event) {
    switch(event.GetId()) {
        case ParameterIds::LR_LOCATION: getProject().lrLocation = (float)event.GetValue(); break;
        case ParameterIds::LR_SH: getProject().lrSh = (float)event.GetValue(); break;
        case ParameterIds::LR_SCALE: getProject().lrScale = (float)event.GetValue(); break;
        case ParameterIds::LR_OPACITY: getProject().lrOpacity = (float)event.GetValue(); break;
        case ParameterIds::LR_ROTATION: getProject().lrRotation = (float)event.GetValue(); break;
        default: break;
    }
}

void UiPanelToolsTrain::onSpinIntervalCapture(wxSpinEvent& event) {
    getProject().intervalCapture = spinIntervalCapture->GetValue();
    refreshText();
}

void UiPanelToolsTrain::onSpinIntervalDensify(wxSpinEvent& event) {
    getProject().intervalDensify = spinIntervalDensify->GetValue();
    refreshText();
}
