#include "UiPanelParamsDensify.h"

Project& UiPanelParamsDensify::getProject() const {
    return *frame.project;
}

UiPanelParamsDensify::UiPanelParamsDensify(wxWindow* parent, UiFrame& frame) : wxPanel(parent), frame(frame) {
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

    sizer->Add(new wxStaticText(this, wxID_ANY, "Cull Opacity Margin"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinParamCullOpacity = new wxSpinCtrlDouble(this, P_CULL_OPACITY);
    spinParamCullOpacity->SetToolTip("Splats with opacities less than this value will be removed. Applied on densify iterations.");
    spinParamCullOpacity->SetRange(0.0f, 100000.0f);
    spinParamCullOpacity->SetDigits(6);
    spinParamCullOpacity->SetIncrement(0.0001);
    spinParamCullOpacity->SetMinSize({96, -1});
    sizer->Add(spinParamCullOpacity, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizer->Add(new wxStaticText(this, wxID_ANY, "Cull Size Margin"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinParamCullSize = new wxSpinCtrlDouble(this, P_CULL_SIZE);
    spinParamCullSize->SetToolTip("Splats with total scale vector length less than this value will be removed. Applied on densify iterations.");
    spinParamCullSize->SetRange(0.0f, 100000.0f);
    spinParamCullSize->SetDigits(6);
    spinParamCullSize->SetIncrement(0.0001);
    spinParamCullSize->SetMinSize({96, -1});
    sizer->Add(spinParamCullSize, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizer->Add(new wxStaticText(this, wxID_ANY, "Densify Variance Trigger"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinParamDensifyVariance = new wxSpinCtrlDouble(this, P_DENSIFY_VARIANCE);
    spinParamDensifyVariance->SetToolTip("Splats with location gradient variance (average of all location gradient magnitudes) "
                                       "greater than this value will be cloned or split. Applied on densify iterations.");
    spinParamDensifyVariance->SetRange(0.0f, 100000.0f);
    spinParamDensifyVariance->SetDigits(6);
    spinParamDensifyVariance->SetIncrement(0.0001);
    spinParamDensifyVariance->SetMinSize({96, -1});
    sizer->Add(spinParamDensifyVariance, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    /*
    sizer->Add(new wxStaticText(this, wxID_ANY, "Densify Size Trigger"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinParamDensifySize = new wxSpinCtrlDouble(this, P_DENSIFY_SIZE);
    spinParamDensifySize->SetToolTip("Splats with total scale vector length greater than this value will be split. Applied on densify iterations.");
    spinParamDensifySize->SetRange(0.0f, 100000.0f);
    spinParamDensifySize->SetDigits(6);
    spinParamDensifySize->SetIncrement(0.0001);
    spinParamDensifySize->SetMinSize({96, -1});
    sizer->Add(spinParamDensifySize, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));*/

    sizer->Add(new wxStaticText(this, wxID_ANY, "Clone/Split Size Margin"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinParamSplitSize = new wxSpinCtrlDouble(this, P_SPLIT_SIZE);
    spinParamSplitSize->SetToolTip("Splats with total scale vector length greater than this value will be split. Otherwise "
                                   "they will be cloned. Applied on densify iterations.");
    spinParamSplitSize->SetRange(0.0f, 100000.0f);
    spinParamSplitSize->SetDigits(6);
    spinParamSplitSize->SetIncrement(0.0001);
    spinParamSplitSize->SetMinSize({96, -1});
    sizer->Add(spinParamSplitSize, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizer->Add(new wxStaticText(this, wxID_ANY, "Split Distance"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinParamSplitDistance = new wxSpinCtrlDouble(this, P_SPLIT_DISTANCE);
    spinParamSplitDistance->SetToolTip("Expressed as a factor of the splat's size. Split splats are displaced along their "
                                       "major scale axis (in both directions) by this amount.");
    spinParamSplitDistance->SetRange(0.0f, 100000.0f);
    spinParamSplitDistance->SetDigits(6);
    spinParamSplitDistance->SetIncrement(0.0001);
    spinParamSplitDistance->SetMinSize({96, -1});
    sizer->Add(spinParamSplitDistance, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizer->Add(new wxStaticText(this, wxID_ANY, "Split New Scale"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinParamSplitScale = new wxSpinCtrlDouble(this, P_SPLIT_SCALE);
    spinParamSplitScale->SetToolTip("The size of new splats resulting from a split, expressed as a factor of the original "
                                    "splat's size.");
    spinParamSplitScale->SetRange(0.0f, 100000.0f);
    spinParamSplitScale->SetDigits(6);
    spinParamSplitScale->SetIncrement(0.0001);
    spinParamSplitScale->SetMinSize({96, -1});
    sizer->Add(spinParamSplitScale, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizer->Add(new wxStaticText(this, wxID_ANY, "Clone Distance"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinParamCloneDistance = new wxSpinCtrlDouble(this, P_CLONE_DISTANCE);
    spinParamCloneDistance->SetToolTip("Expressed as a factor of the splat's size. Cloned splats are displaced in the direction "
                                       "of their location gradient by this amount.");
    spinParamCloneDistance->SetRange(0.0f, 100000.0f);
    spinParamCloneDistance->SetDigits(6);
    spinParamCloneDistance->SetIncrement(0.0001);
    spinParamCloneDistance->SetMinSize({96, -1});
    sizer->Add(spinParamCloneDistance, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));

    Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelParamsDensify::onSpinParameter, this);

    SetSizerAndFit(sizer);
}

void UiPanelParamsDensify::refreshProject() {
    spinParamCullOpacity->SetValue(getProject().paramCullOpacity);
    spinParamCullSize->SetValue(getProject().paramCullSize);
    spinParamDensifyVariance->SetValue(getProject().paramDensifyVariance);
    //spinParamDensifySize->SetValue(getProject().paramDensifySize);
    spinParamSplitSize->SetValue(getProject().paramSplitSize);
    spinParamSplitDistance->SetValue(getProject().paramSplitDistance);
    spinParamSplitScale->SetValue(getProject().paramSplitScale);
    spinParamCloneDistance->SetValue(getProject().paramCloneDistance);
}

void UiPanelParamsDensify::onSpinParameter(wxSpinDoubleEvent& event) {
    switch(event.GetId()) {
        case ParamIds::P_CULL_OPACITY: getProject().paramCullOpacity = (float)event.GetValue(); break;
        case ParamIds::P_CULL_SIZE: getProject().paramCullSize = (float)event.GetValue(); break;
        case ParamIds::P_DENSIFY_VARIANCE: getProject().paramDensifyVariance = (float)event.GetValue(); break;
        //case ParamIds::P_DENSIFY_SIZE: getProject().paramDensifySize = (float)event.GetValue(); break;
        case ParamIds::P_SPLIT_SIZE: getProject().paramSplitSize = (float)event.GetValue(); break;
        case ParamIds::P_SPLIT_DISTANCE: getProject().paramSplitDistance = (float)event.GetValue(); break;
        case ParamIds::P_SPLIT_SCALE: getProject().paramSplitScale = (float)event.GetValue(); break;
        case ParamIds::P_CLONE_DISTANCE: getProject().paramCloneDistance = (float)event.GetValue(); break;
        default: break;
    }
}
