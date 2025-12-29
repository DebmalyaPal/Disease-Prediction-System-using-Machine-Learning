import React, { useEffect, useState } from "react";
import {
	Container,
	Typography,
	TextField,
	MenuItem,
	Button,
	Card,
	CardContent,
	FormControl,
	InputLabel,
	Select,
	OutlinedInput,
	Checkbox,
	ListItemText,
	Box,
	Alert
} from "@mui/material";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import axios from "axios";

const API_BASE = "";

function App() {
	const [symptoms, setSymptoms] = useState([]);
	const [selectedSymptoms, setSelectedSymptoms] = useState([]);
	const [age, setAge] = useState("");
	const [gender, setGender] = useState("");
	const [predictions, setPredictions] = useState([]);
	const [warning, setWarning] = useState("");

	// Fetch symptoms on startup
	useEffect(() => {
		axios
			.get(`${API_BASE}/symptoms`)
			.then((res) => {
				setSymptoms(res.data.symptoms || []);
			})
			.catch((err) => console.error("Error fetching symptoms: ", err));
	}, []);

	const handleSubmit = () => {
		if (selectedSymptoms.length === 0) {
			toast.error("Please select at least one symptom.");
			return;
		}

		setWarning("");

		const payload = {
			symptoms: Object.fromEntries(selectedSymptoms.map((s) => [s, 1]))
		};

		axios
			.post(`${API_BASE}/predict`, payload)
			.then((res) => {
				if (res.data.success) {
					setPredictions(res.data.predictions);
				}
			})
			.catch((err) => console.error("Prediction error: ", err));
	};

	const handleReset = () => {
		setAge("");
		setGender("");
		setSelectedSymptoms([]);
		setPredictions([]);
		setWarning("");
	};

	return (
		<Box
			sx={{
				minHeight: "100vh",
				background: "linear-gradient(135deg, #e3f2fd, #bbdefb)",
				paddingTop: 4,
				paddingBottom: 4
			}}
		>
			<Container maxWidth="md">
				{/* Heading */}
				<Typography
					variant="h4"
					align="center"
					sx={{ fontWeight: "bold", color: "#0d47a1", marginBottom: 3 }}
				>
					MediPredict — ML Powered Diagnostic Engine
				</Typography>

				{/* Input Card */}
				<Card
					elevation={6}
					sx={{
						padding: 3,
						borderRadius: 4,
						background: "rgba(255,255,255,0.9)",
						backdropFilter: "blur(6px)"
					}}
				>
					<CardContent>
						<Typography variant="h6" align="center" sx={{ marginBottom: 2 }}>
							Enter Patient Details
						</Typography>

						{warning && (
							<Alert severity="error" sx={{ marginBottom: 2 }}>
								{warning}
							</Alert>
						)}

						<Box sx={{ display: "flex", gap: 2, marginBottom: 2 }}>
							<TextField
								label="Age"
								type="number"
								fullWidth
								value={age}
								onChange={(e) => {
									const value = e.target.value;

									// Check if value is empty or not a number
									if (value === "" || isNaN(value)) {
										toast.error("Please enter a valid number for age.");
										setAge("");
										return;
									}
									const num = Number(value);
									// Range validation
									if (num < 0 || num > 100) {
										toast.error("Please select a valid age between 0 and 100.");
										setAge(0);
										return;
									}
									setAge(num);
								}}
								inputProps={{ min: 0, max: 100, step: 1 }}
							/>

							<TextField
								select
								label="Gender"
								fullWidth
								value={gender}
								onChange={(e) => setGender(e.target.value)}
							>
								<MenuItem value="male">Male</MenuItem>
								<MenuItem value="female">Female</MenuItem>
								<MenuItem value="other">Other</MenuItem>
							</TextField>
						</Box>

						{/* Symptom Multi-select */}
						<FormControl fullWidth>
							<InputLabel>Symptoms</InputLabel>
							<Select
								multiple
								value={selectedSymptoms}
								onChange={(e) => setSelectedSymptoms(e.target.value)}
								input={<OutlinedInput label="Symptoms" />}
								renderValue={(selected) =>
											selected
											.map(code => symptoms.find(s => s.code === code)?.name)
											.join(", ")
										}
							>
								{symptoms.map((sym) => (
									<MenuItem key={sym.id} value={sym.code}>
										<Checkbox checked={selectedSymptoms.includes(sym.code)} />
										<ListItemText primary={sym.name} />
									</MenuItem>
								))}
							</Select>
						</FormControl>

						{/* Buttons */}
						<Box sx={{ display: "flex", gap: 2, marginTop: 3 }}>
							<Button
								variant="contained"
								color="primary"
								fullWidth
								onClick={handleSubmit}
								sx={{ fontWeight: "bold" }}
							>
								Predict
							</Button>

							<Button
								variant="outlined"
								color="secondary"
								fullWidth
								onClick={handleReset}
								sx={{ fontWeight: "bold" }}
							>
								Reset
							</Button>
						</Box>
					</CardContent>
				</Card>

				{/* Results */}
				{predictions.length > 0 && (
					<Card elevation={4} sx={{ padding: 3, borderRadius: 3, marginTop: 3 }}>
						<Typography
							variant="h5"
							align="center"
							fontWeight={"bold"}
							sx={{ marginBottom: 2 }}
						>
							Diagnostic Summary & Recommendations
						</Typography>

						{predictions.map((p, idx) => (
							<Accordion key={idx} sx={{ marginBottom: 1 }}>
								<AccordionSummary
									expandIcon={<ExpandMoreIcon />}
									aria-controls={`panel${idx}-content`}
									id={`panel${idx}-header`}
								>
									<Typography variant="h6" sx={{ color: "#1976d2" }}>
										{p.name} — {p.probability}
									</Typography>
								</AccordionSummary>

								<AccordionDetails>
									<Typography variant="body1" sx={{ marginBottom: 1 }}>
										{p.description}
									</Typography>

									<Typography variant="subtitle1">Precautions:</Typography>
									<ul>
										{p.precautions.map((pr, i) => (
											<li key={i}>{pr}</li>
										))}
									</ul>
								</AccordionDetails>
							</Accordion>
						))}
					</Card>
				)}


				{/* Footer */}
				<Typography
					variant="body2"
					align="center"
					sx={{ marginTop: 4, opacity: 0.7 }}
				>
					© 2022 MediPredict • UEM Kolkata • Developed by Debmalya Pal
				</Typography>
			</Container>

			<ToastContainer position="top-right" autoClose={3000} />

		</Box>
	);
}

export default App;
