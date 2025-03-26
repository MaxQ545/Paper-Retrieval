// src/api.js
import axios from 'axios';

const API_URL = 'http://172.16.3.134:5000/search';

export const fetchPapers = async (query) => {
  try {
    console.log('Send Request:');  // Debugging line
    const response = await axios.get(API_URL, {
      params: { query },
      headers: {
        'Access-Control-Allow-Origin': '*'  // This is usually handled by the server
      }
    });
    console.log('API Response:', response.data);  // Debugging line
    return response.data;
  } catch (error) {
    console.error('Error fetching papers:', error);
    return [];
  }
};
