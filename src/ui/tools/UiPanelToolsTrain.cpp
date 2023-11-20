#include "UiPanelToolsTrain.h"

#include <wx/progdlg.h>

#include "ui/UiPanelTools.h"
#include "ui/UiPanelViewOutput.h"
#include "Trainer.cuh"

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



    wxBoxSizer* sizerManual = new wxBoxSizer(wxVERTICAL);
    sizer->Add(sizerManual, wxSizerFlags().Border());
    sizerManual->Add(new wxStaticText(this, wxID_ANY, "Manual Train"));

    wxBoxSizer* sizerManualH = new wxBoxSizer(wxHORIZONTAL);
    sizerManual->Add(sizerManualH);

    wxBoxSizer* sizerManual1 = new wxBoxSizer(wxVERTICAL);
    sizerManualH->Add(sizerManual1);
    wxBoxSizer* sizerManual2 = new wxBoxSizer(wxVERTICAL);
    sizerManualH->Add(sizerManual2);

    buttonManual1 = new wxButton(this, M_1, "1x");
    sizerManual1->Add(buttonManual1);
    buttonManual5 = new wxButton(this, M_5, "5x");
    sizerManual1->Add(buttonManual5);
    buttonManual10 = new wxButton(this, M_10, "10x");
    sizerManual1->Add(buttonManual10);
    buttonManual20 = new wxButton(this, M_20, "20x");
    sizerManual1->Add(buttonManual20);
    buttonManual50 = new wxButton(this, M_50, "50x");
    sizerManual2->Add(buttonManual50);
    buttonManual100 = new wxButton(this, M_100, "100x");
    sizerManual2->Add(buttonManual100);
    buttonManual200 = new wxButton(this, M_200, "200x");
    sizerManual2->Add(buttonManual200);
    buttonManualDensify = new wxButton(this, M_D, "1x Densify");
    sizerManual2->Add(buttonManualDensify);

    Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsTrain::onButtonManual, this);



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



    SetSizerAndFit(sizer);
}

void UiPanelToolsTrain::refreshProject() {
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

    buttonManual1->Disable();
    buttonManual5->Disable();
    buttonManual10->Disable();
    buttonManual20->Disable();
    buttonManual50->Disable();
    buttonManual100->Disable();
    buttonManual200->Disable();
    buttonManualDensify->Disable();
}

void UiPanelToolsTrain::onButtonAutoStop(wxCommandEvent& event) {
    getFrame().autoTraining = false;
    buttonAutoStart->Enable();
    buttonAutoStop->Disable();

    buttonManual1->Enable();
    buttonManual5->Enable();
    buttonManual10->Enable();
    buttonManual20->Enable();
    buttonManual50->Enable();
    buttonManual100->Enable();
    buttonManual200->Enable();
    buttonManualDensify->Enable();
}

void UiPanelToolsTrain::onButtonManual(wxCommandEvent& event) {
    bool isManualButton = false;
    for (int i = ManualIds::M_1; i < ManualIds::M_END; i++) {
        if (event.GetId() == i) {
            isManualButton = true;
            break;
        }
    }
    if (!isManualButton) return;

    int count = 1;
    bool densify = false;
    switch(event.GetId()) {
        case ManualIds::M_5: count = 5; break;
        case ManualIds::M_10: count = 10; break;
        case ManualIds::M_20: count = 20; break;
        case ManualIds::M_50: count = 50; break;
        case ManualIds::M_100: count = 100; break;
        case ManualIds::M_200: count = 200; break;
        case ManualIds::M_D: densify = true; break;
        default: break;
    }

    if (count > 1) {
        wxProgressDialog dialog("Training Gaussian Splats", "Training for " + std::to_string(count) + " iterations...",
                                count, getFrame().panelOutput, wxPD_AUTO_HIDE | wxPD_CAN_ABORT);
        for (int i = 0; i < count; i++) {
            getFrame().trainer->train(getProject(), densify);
            if(!dialog.Update(i + 1)) break;
        }
    } else getFrame().trainer->train(getProject(), densify);

    getFrame().panelOutput->refreshText();
    refreshText();
}

void UiPanelToolsTrain::onSpinIntervalCapture(wxSpinEvent& event) {
    getProject().intervalCapture = spinIntervalCapture->GetValue();
    refreshText();
}

void UiPanelToolsTrain::onSpinIntervalDensify(wxSpinEvent& event) {
    getProject().intervalDensify = spinIntervalDensify->GetValue();
    refreshText();
}
