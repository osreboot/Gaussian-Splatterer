#include "UiPanelParams.h"

#include "UiFrame.h"
#include "Project.h"

Project& UiPanelParams::getProject() const {
    return *dynamic_cast<UiFrame*>(GetParent()->GetParent())->project;
}

UiPanelParams::UiPanelParams(wxWindow *parent) : wxPanel(parent) {
    sizer = new wxBoxSizer(wxVERTICAL);

    panelLr = new UiPanelParamsLr(this);
    sizer->Add(panelLr, wxSizerFlags().Expand().Border());

    panelDensify = new UiPanelParamsDensify(this);
    sizer->Add(panelDensify, wxSizerFlags().Expand().Border());

    panelOther = new UiPanelParamsOther(this);
    sizer->Add(panelOther, wxSizerFlags().Expand().Border());

    SetSizerAndFit(sizer);
}

void UiPanelParams::refreshProject() {
    panelLr->refreshProject();
    panelDensify->refreshProject();
    panelOther->refreshProject();
}
