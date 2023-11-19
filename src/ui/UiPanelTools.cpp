#include "UiPanelTools.h"

#include "UiFrame.h"

#include "Project.h"

Project& UiPanelTools::getProject() const {
    return *dynamic_cast<UiFrame*>(GetParent()->GetParent())->project;
}

UiPanelTools::UiPanelTools(wxWindow *parent) : wxPanel(parent) {
    sizer = new wxBoxSizer(wxHORIZONTAL);

    sizerStaticInput = new wxStaticBoxSizer(wxVERTICAL, this, "1. Input Model Data");
    sizer->Add(sizerStaticInput, wxSizerFlags().Expand().Border());

    panelTruth = new UiPanelToolsTruth(this);
    sizer->Add(panelTruth, wxSizerFlags().Expand().Border());

    panelTrain = new UiPanelToolsTrain(this);
    panelTrain->Disable();
    sizer->Add(panelTrain, wxSizerFlags().Expand().Border());

    panelView = new UiPanelToolsView(this);
    sizer->Add(panelView, wxSizerFlags().Expand().Border());

    SetSizerAndFit(sizer);
}

void UiPanelTools::refreshProject() {
    panelTruth->refreshProject();
    panelTrain->refreshProject();
    panelView->refreshProject();
}
